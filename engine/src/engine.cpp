// rapidli-engine — Rapidli's serving loop for GGUF chat models on top of a patched llama.cpp, built for total throughput.
//
// The stock server gives every new prompt its own pass over the model (a different graph, never pre-recorded on the
// card), and every such pass stalls all other streams. Here every pass has the same shape: all lanes, one token each.
// A lane that reads a prompt takes one prompt token per pass while the other lanes keep generating, so the graph is
// built once, stays recorded on the card, and every stream's recurrent state is updated in place.
//   - prompts longer than --trickle tokens are read in passes of their own (--chunk tokens per pass), one such pass
//     between two ordinary passes, so the other lanes keep generating
//   - with fewer requests the pass narrows to 1, 2, 4, 6, 8, 12, ... 128 or 144 lanes (one recorded graph per width)
//   - one sampling operation for all lanes on the card; each request chooses with "temperature" (0 = greedy, the
//     default, or --temp), "min_p" (default 0.05: tokens below 5% of the most likely token's probability take no part)
//     and "seed". There is no top-k / top-p in this mode; --chains 1 uses llama.cpp's sampler chains instead (slower,
//     the same settings for every request)
//   - --state 8 keeps the recurrent state as 8-bit codes instead of 16-bit floats: faster and 3 GB smaller at 80 lanes,
//     but the answers change (measured quality equal on our 300 questions; see RESULTS.md)
//
//   rapidli-engine --model <gguf> [--lanes 64] [--ctx 384] [--port 11832] [--threads 4] [--trickle 96] [--chunk 512] [--kvu 1] [--state 16|8] [--temp 0] [--min-p 0.05] [--chains 0 [--top-k 40] [--top-p 0.95] [--seed 1]]
// HTTP (blocking, one request per connection thread):
//   POST /v1/chat/completions {"messages": [...], "max_tokens": n, "temperature": 0, "min_p": 0.05, "seed": 1, "stream": false,
//                              "chat_template_kwargs": {"enable_thinking": false}}
//                             ("stream": true answers with server-sent events, one chunk per pass)
//   POST /completion          {"prompt": "<already templated text>", "n_predict": n}
//   POST /warm                {"ms": n}: the card brought up to speed while the user types (no lane; see Warmer)
//   GET  /health, GET /stats
#include "llama.h"
#include "ggml-backend.h"
#include "cpp-httplib/httplib.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <functional>
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "ggml-alloc.h"

// Rapidli: the card warmed while the user types (POST /warm {"ms": n}). After a few seconds without work the card lowers
// its clocks, and the first prompt read after that took 0.65 s instead of 0.39 (20 Nov). This is a matrix multiplication
// on a backend of its own (its own stream, about 200 MB of the card, made at the first /warm), repeated until the time is up or a
// request comes in (enqueue cancels it): no lane, no model state, nothing the lanes or the stats show.
struct Warmer {
    std::mutex              mu;
    std::condition_variable cv;
    std::once_flag          once;
    std::atomic<int64_t>    until_ms{0};
    bool                    failed = false;
    ggml_backend_t          be  = nullptr;
    ggml_context *          ctx = nullptr;
    ggml_backend_buffer_t   buf = nullptr;
    ggml_cgraph *           gf  = nullptr;

    static int64_t now() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    bool init() {
        if (be || failed) { return be != nullptr; }
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!be) { failed = true; return false; }
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        // (a multiplication, which keeps the card's compute busy: an elementwise pass over 32 MB, memory-bound, did not
        // bring the clocks up, 0.63 s after it against 0.39 s; either way the backend takes about 190 MB)
        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4096, 4096);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4096, 1024);
        ggml_tensor * c = ggml_mul_mat(ctx, a, b);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, c);
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!buf) { failed = true; ggml_backend_free(be); be = nullptr; return false; }
        ggml_backend_buffer_clear(buf, 0);
        return true;
    }
    void run() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&] { return until_ms.load() > now(); });
            lk.unlock();
            if (init()) {
                while (until_ms.load() > now()) { ggml_backend_graph_compute(be, gf); }
            } else {
                until_ms = 0;
            }
            lk.lock();
        }
    }
    void warm(int ms) {
        std::call_once(once, [this] { std::thread([this] { run(); }).detach(); });
        { std::lock_guard<std::mutex> lk(mu); until_ms = now() + std::max(0, std::min(ms, 2000)); }
        cv.notify_one();
    }
    void cancel() { until_ms = 0; }
};
static Warmer & g_warmer = *new Warmer();   // never destroyed: its thread may still wait at exit

using json = nlohmann::ordered_json;

// our llama.dll: the sampling settings of a sequence for the one fused sampling operation (engine/llama/patch_greedy.py);
// temperature 0 = arg max; min_p = tokens below this fraction of the largest probability take no part
#if defined(_WIN32)
#  define FQ_IMPORT __declspec(dllimport)
#else
#  define FQ_IMPORT
#endif
extern "C" FQ_IMPORT void llama_fq_set_sampling(int32_t seq, float temperature, float min_p, uint32_t seed);
// version 2 (engine/llama/patch_zfork.py): the whole memory of a sequence copied to other sequences, on the card
extern "C" FQ_IMPORT int32_t llama_fq_fork(llama_context * ctx, int32_t seq_src, const int32_t * seq_dst, int32_t n_dst);

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// version 2: the branches of one fork report to one connection
struct Request;
struct Group {
    std::vector<std::weak_ptr<Request>> members;   // for races: the first branch of a race to end properly stops the others
    std::mutex               mu;
    std::condition_variable  cv;
    std::vector<std::string> fresh;
    std::vector<json>        finals;
    int                      n_done = 0;
    explicit Group(size_t n) : fresh(n), finals(n) {}
};

struct Request {
    std::vector<llama_token> prompt;
    // version 2
    int                      fork_src  = -1;     // start from this parked state (session) instead of an empty lane
    bool                     hold      = false;  // park the lane's state when the request ends; the answer names the session
    bool                     bulk      = false;  // read the prompt in passes of its own however short it is
    bool                     read_only = false;  // read the prompt, write nothing
    std::vector<std::string> stops;              // the answer ends before the first of these strings
    std::shared_ptr<Group>   group;              // a branch of a fork: text and result go to the group
    int                      branch    = 0;
    std::string              name;               // of the session this request parks (hold)
    int                      soft      = 0;      // after this many written tokens the answer ends at the next end of a sentence
    int                      loop      = 0;      // the answer ends ("loop") when its last this many tokens were written before, in a row
    int                      race      = -1;     // branches of one fork with the same number write the same thing: first wins
    int                      max_tokens = 200;
    double                   t_arrive   = 0;
    std::promise<json>       done;
    std::atomic<bool>        cancelled{false};   // the client went away: the lane is freed at the next pass
    float                    temp = 0.0f, min_p = 0.05f;   // sampling of this request (temperature 0 = greedy)
    uint32_t                 seed = 0;

    // streaming: the engine thread appends the text of every token, the connection thread takes it away
    bool                     stream = false;
    std::mutex               smu;
    std::condition_variable  scv;
    std::string              fresh;              // text not yet taken by the connection thread
    bool                     sdone = false;
    json                     final;

    void push(const std::string & piece) {
        if (group) {
            { std::lock_guard<std::mutex> lk(group->mu); group->fresh[branch] += piece; }
            group->cv.notify_one();
            return;
        }
        { std::lock_guard<std::mutex> lk(smu); fresh += piece; }
        scv.notify_one();
    }
};

// number of leading bytes of s that end on a character boundary (a token can end inside a multi-byte character)
static size_t utf8_whole(const std::string & s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= 3 && back <= n; ++back) {
        const unsigned char c = (unsigned char) s[n - back];
        if ((c & 0xC0) == 0x80) { continue; }                                    // a continuation byte: look further back
        const size_t need = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        return need > back ? n - back : n;
    }
    return n;
}

struct Lane {
    std::shared_ptr<Request> req;
    size_t      n_fed   = 0;      // prompt tokens already given to the model
    int         n_gen   = 0;      // tokens generated
    llama_pos   pos     = 0;
    llama_token last    = 0;
    bool        clear   = true;   // the sequence must be emptied before its next token
    bool        expects = false;  // the token fed in this pass is followed by a generated token
    bool        samp    = false;  // the card draws at random for this lane (switched off again when the lane is idle)
    std::string text;
    double      t_start = 0, t_prompt = 0;
    int         idx     = 0;      // the lane's number = its sequence
    int         own     = 0;      // tokens fed since the lane was emptied or forked (its own cells of the attention cache)
    bool        failed  = false;  // the fork for this lane did not work
    std::string head;             // the beginning of what the lane was given to read (for the test bench)
    int         fed_bytes = -1;   // after a stop string: bytes of the written text that the model has read (the last token was not)
    std::vector<llama_token> toks; // the tokens written (for the loop check of requests that ask for one)
};

// version 2: one answer drafted by the model's own multi-token-prediction head
struct Spec {
    llama_context *   tgt = nullptr;   // the full model, one sequence, rollback snapshots for n_draft tokens
    llama_context *   dft = nullptr;   // the MTP head
    llama_batch       tb{};
    llama_batch_ext * db = nullptr;
    int               n_draft = 0, n_embd = 0, n_vocab = 0, n_batch = 64;
    // version 3: drafts copied from the text itself (prompt lookup): when the last tokens occurred before, what followed
    // them then is checked as the draft. Up to n_lookup tokens; works without an MTP head (dft null: lookup only).
    int               n_lookup = 0;
    std::mutex        busy;            // one drafted answer at a time
    std::atomic<long long> n_answers{0}, n_tokens{0}, n_cycles{0}, n_kept{0}, n_drafted{0};
    std::atomic<long long> n_lk_cycles{0}, n_lk_kept{0}, n_lk_drafted{0};
    // what the choice between the drafts is made from, measured while answering and kept between answers:
    //   pass_ms[r]: the full model reading r rows at once (r = 1 .. 32), draft_ms: one step of the MTP head,
    //   mtp_acc[k]: how often the MTP head's k-th draft is kept when the ones before it were,
    //   lk_acc[n]: how often a copied token is kept when the copy matched the last n tokens (n = 1 .. 8)
    double            pass_ms[33] = {0}, draft_ms = 0.0, mtp_acc[17] = {0}, lk_acc[9] = {0};
    // two parked prefixes (e.g. the layout texts read before the question); the one used least recently is replaced
    struct Prefix { std::string text; std::vector<uint8_t> tgt, dft; std::vector<float> pend; int len = 0; long long used = 0; };
    Prefix               px[4];       // (19 Nov: 4, was 2: the two layouts and the conversation were evicting each other)
    long long            px_clock = 0;
    std::atomic<long long> n_px_hits{0};
    // what the drafted answer is doing now (for GET /lanes)
    std::mutex        view_mu;
    bool              active = false;
    int               v_tokens = 0, v_kept = 0, v_drafted = 0;
    std::string       v_head, v_tail;
};

static llama_token argmax_row(const float * lg, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i) { if (lg[i] > lg[b]) { b = i; } }
    return b;
}

struct Engine {
    llama_model *       model = nullptr;
    llama_context *     ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    int                 lanes = 64, ctx_lane = 384;
    int                 ctx_extra = 0;    // version 2: cells for long parked prompts (a chat's history)
    int                 trickle = 96;     // prompts up to this many tokens are fed one token per pass ...
    bool                alone_bulk = true; // ... unless their lane is the only one with a request (--alone-bulk 0: always)
    int                 n_busy = 0;       // lanes with a request, counted before every pass
    int                 narrow_wait = 8;  // passes in a row a narrower width would have covered before the pass narrows (--narrow-wait;
                                          // 48 before 20 Nov: 150 requests of 60-250 tokens took 14.6 s, with 8 14.3 s)
    int                 few_bulk = 8;     // ... "alone": this many lanes with a request or fewer (--few-bulk; a bulk pass costs the
                                          // others one pass and saves the reader one pass per token: 100 lanes reading
                                          // their tags together still read them side by side)
    int                 chunk   = 512;    // longer prompts: tokens per pass of their own
    std::vector<int>    widths;           // allowed numbers of lanes in a pass, ascending, the last one = lanes
    std::atomic<int>    width{0};
    llama_token         filler = 0;
    bool                fused  = true;    // one sampling operation for all lanes, settings per request
    float               def_temp = 0.0f, def_min_p = 0.05f;
    std::atomic<uint32_t> seed_next{1};

    // version 2: parked states ("sessions"). Sequence lanes + k holds session k; it takes part in no pass, so it stays
    // as it was parked. A request that names a session starts with a copy of it (llama_fq_fork) and reads only its own
    // text; many requests can start from one session at once: that is the fork.
    struct Slot {
        bool        used = false, user = false, freeing = false;
        int         refs = 0;          // requests that will start from it and have not yet
        llama_pos   pos = 0;
        llama_token pending = -1;      // the last token written before parking, not yet read by the model
        std::string name;              // given by the client: a later request with the same name gets this session back
    };
    std::vector<Slot>   vault;

    // version 2: what the lanes are doing, published after every pass for GET /lanes
    struct Snap { bool on = false, fork = false, reader = false; int n_gen = 0, n_prompt = 0; std::string head, tail; };
    std::mutex          snap_mu;
    std::vector<Snap>   snap;
    std::deque<json>    recent;        // the last requests that ended
    std::atomic<int>    n_sessions{0};
    std::atomic<bool>   wake{false};
    std::mutex          pass_mu;       // one pass at a time on the card: the lanes' passes and those of /spec take turns
    // version 2: work that must run on the run thread between passes (reading an image into a parked session)
    std::mutex                          job_mu;
    std::deque<std::function<void()>>   jobs;
    mtmd_context *                      mctx = nullptr;   // the vision projector (--mmproj), or none
    void run_jobs() {
        std::deque<std::function<void()>> todo;
        { std::lock_guard<std::mutex> lk(job_mu); todo.swap(jobs); }
        for (auto & j : todo) { j(); }
    }

    bool release(int s) {
        {
            std::lock_guard<std::mutex> lk(mu);
            if (s < 0 || s >= (int) vault.size() || !vault[s].used) { return false; }
            vault[s].user = false;
            if (vault[s].refs <= 0) { vault[s].freeing = true; }
        }
        wake = true;
        cv.notify_one();
        return true;
    }

    std::mutex                           mu;
    std::condition_variable              cv;
    std::deque<std::shared_ptr<Request>> queue;
    std::atomic<bool>                    stop{false};

    // statistics
    std::atomic<long long> n_pass{0}, n_tokens{0}, n_prompt{0}, n_done{0}, n_prefill{0}, n_forks{0}, n_moves{0};
    int                    pack = 1;          // version 2: lanes move down when a narrower pass could hold them (--pack 0 = off)
    std::atomic<double>    ms_pass{0};

    std::string piece(llama_token t) const {
        char buf[256];
        const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
        return n > 0 ? std::string(buf, n) : std::string();
    }

    std::vector<llama_token> tokenize(const std::string & s) const {
        std::vector<llama_token> out(s.size() + 16);
        int n = llama_tokenize(vocab, s.data(), (int32_t) s.size(), out.data(), (int32_t) out.size(), true, true);
        if (n < 0) {
            out.resize(-n);
            n = llama_tokenize(vocab, s.data(), (int32_t) s.size(), out.data(), (int32_t) out.size(), true, true);
        }
        out.resize(n > 0 ? n : 0);
        return out;
    }

    void finish(Lane & L, const char * why) {
        const double t  = now_ms();
        const double dt = t - L.t_prompt;
        json r = {
            {"content", L.text},
            {"stop", why},
            {"tokens_predicted", L.n_gen},
            {"tokens_evaluated", (int) L.req->prompt.size()},
            {"timings", {
                {"queue_ms", L.t_start - L.req->t_arrive},
                {"prompt_ms", L.t_prompt - L.t_start},
                {"predicted_ms", dt},
                {"predicted_per_second", dt > 0 ? 1000.0 * L.n_gen / dt : 0.0},
            }},
        };
        const std::string reason = why;
        {
            std::lock_guard<std::mutex> lk(snap_mu);
            recent.push_back(json{{"lane", L.idx}, {"head", L.head}, {"text", L.text.substr(0, 120)}, {"tokens", L.n_gen}, {"read", (int) L.req->prompt.size()},
                                  {"ms", now_ms() - L.t_start}, {"stop", reason}, {"fork", L.req->fork_src >= 0}, {"n", (long long) n_done}});
            if (recent.size() > 60) { recent.pop_front(); }
        }
        // version 2: how much of the written text is in the lane's state (a continuation has to supply the rest)
        r["fed_bytes"] = L.fed_bytes >= 0 ? L.fed_bytes : (int) L.text.size();
        L.fed_bytes = -1;
        if (L.req->hold && reason != "error" && reason != "cancelled") {
            int sl = -1;
            {
                std::lock_guard<std::mutex> lk(mu);
                for (size_t k = 0; k < vault.size(); ++k) { if (!vault[k].used) { sl = (int) k; break; } }
                if (sl >= 0) {
                    vault[sl] = Slot();
                    vault[sl].used = true; vault[sl].user = true; vault[sl].pos = L.pos;
                    vault[sl].pending = reason == "limit" ? L.last : -1;
                    vault[sl].name = L.req->name;
                }
            }
            if (sl >= 0) {
                const int32_t d = lanes + sl;
                if (llama_fq_fork(ctx, L.idx, &d, 1) != 0) { n_sessions++; }
                else { std::lock_guard<std::mutex> lk(mu); vault[sl] = Slot(); sl = -1; }
            }
            r["session"] = sl;
        }
        // the lane's own cells are given back now (a parked copy keeps what it shares)
        llama_memory_seq_rm(llama_get_memory(ctx), L.idx, -1, -1);
        L.pos = 0; L.own = 0;
        if (L.req->group) {
            {
                std::lock_guard<std::mutex> lk(L.req->group->mu);
                if (L.req->race >= 0 && (reason == "eos" || reason == "stop" || reason == "soft")) {
                    for (auto & w : L.req->group->members) {
                        auto o = w.lock();
                        if (o && o.get() != L.req.get() && o->race == L.req->race) { o->cancelled = true; }
                    }
                }
                L.req->group->finals[L.req->branch] = r; L.req->group->n_done++;
            }
            L.req->group->cv.notify_all();
        } else if (L.req->stream) {
            { std::lock_guard<std::mutex> lk(L.req->smu); L.req->final = r; L.req->sdone = true; }
            L.req->scv.notify_one();
        }
        L.req->done.set_value(std::move(r));
        L.req.reset();
        n_done++;
    }

    // the narrowest allowed pass that covers lanes 0..need-1 (each width has its own recorded graph)
    int width_for(int need) const {
        for (int w : widths) { if (w >= need) { return w; } }
        return lanes;
    }

    // A long prompt is read in runs of up to `chunk` tokens in passes of its own (all but its last tokens), one such
    // pass between two ordinary passes so that the other lanes keep generating. In the ordinary pass a lane that is
    // still reading takes one more prompt token, as a short prompt does.
    size_t bulk_left(const Lane & l) const {
        if (!l.req) { return 0; }
        const size_t left = l.req->prompt.size() - l.n_fed;
        if (l.req->read_only) { return left; }              // version 2: nothing is written after it, so all of it
        if (l.req->bulk) { return left > 2 ? left - 1 : 0; }
        if ((int) l.req->prompt.size() <= trickle) {
            // (fed one token per pass, a lane alone read a 45-token prompt in 45 passes: 0.55 s on Ternary-Bonsai-2-27B; in one
            // bulk pass it takes one. With other lanes at work, one token per pass keeps them generating, 19 Nov 2026)
            // (and only when it has more left than two passes for each lane at work: bulk passes go one lane at a time,
            // so 8 parts of 3 tokens each, read in turn, started staggered: 2.50 s for 8 x 120 tokens against 2.43
            // read one token a pass together, 20 Nov)
            return (alone_bulk && n_busy <= few_bulk && left > (size_t) std::max(2, 2 * n_busy)) ? left - 1 : 0;
        }
        return left > 24 ? left - 1 : 0;                    // a pass of its own is not worth it for the last few tokens
    }

    bool bulk_pass(llama_batch & batch, llama_memory_t mem, int i, Lane & l) {
        const auto & p = l.req->prompt;
        if (l.clear) { llama_memory_seq_rm(mem, i, -1, -1); l.pos = 0; l.own = 0; l.clear = false; }
        const int n = (int) std::min<size_t>(chunk, bulk_left(l));
        l.own += n;
        batch.n_tokens = n;
        for (int k = 0; k < n; ++k) {
            batch.token[k]     = p[l.n_fed++];
            batch.pos[k]       = l.pos++;
            batch.n_seq_id[k]  = 1;
            batch.seq_id[k][0] = i;
            batch.logits[k]    = 0;
        }
        {
            std::lock_guard<std::mutex> pk(pass_mu);
            if (llama_decode(ctx, batch) != 0) { return false; }
        }
        n_prompt += n;
        n_prefill++;
        return true;
    }

    void run() {
        std::vector<Lane> L(lanes);
        llama_batch batch = llama_batch_init(std::max(lanes, chunk), 0, 1);
        llama_memory_t mem = llama_get_memory(ctx);
        bool warned = false;
        int  narrow = 0;          // passes in a row that a narrower width would have covered
        int  rr     = 0;          // the lane after the one that had the last bulk pass
        int  calm   = 0;          // passes in a row in which packing the lanes down would have paid
        { std::lock_guard<std::mutex> lk(snap_mu); snap.assign(lanes, Snap()); }
        auto publish = [&]() {
            std::lock_guard<std::mutex> lk(snap_mu);
            for (int i = 0; i < lanes; ++i) {
                Snap & p = snap[i]; const Lane & l = L[i];
                p.on = (bool) l.req;
                if (!p.on) { continue; }
                p.n_gen = l.n_gen; p.n_prompt = (int) l.req->prompt.size(); p.fork = l.req->fork_src >= 0; p.reader = l.req->read_only;
                p.head = l.head; p.tail = l.text.size() > 110 ? l.text.substr(l.text.size() - 110) : l.text;
            }
        };
        width = widths.front();
        for (int i = 0; i < lanes; ++i) { L[i].idx = i; }

        while (!stop) {
            run_jobs();
            int need = 0;
            {
                std::unique_lock<std::mutex> lk(mu);
                for (size_t k = 0; k < vault.size(); ++k) {     // sessions nobody needs any more
                    if (vault[k].used && vault[k].freeing && vault[k].refs <= 0) {
                        llama_memory_seq_rm(mem, lanes + (int) k, -1, -1);
                        vault[k] = Slot();
                        n_sessions--;
                    }
                }
                bool active = false;
                for (auto & l : L) { active = active || l.req; }
                if (!active && queue.empty()) {
                    cv.wait(lk, [&] { return stop || !queue.empty() || wake; });
                    if (stop) { break; }
                    if (wake.exchange(false) && queue.empty()) { continue; }
                }
                std::vector<std::pair<int, int>> forks;     // (session, lane)
                for (auto & l : L) {                       // the lowest free lanes first, so that the pass stays narrow
                    if (!l.req && !queue.empty()) {
                        l.req = queue.front();
                        queue.pop_front();
                        l.n_fed = 0; l.n_gen = 0; l.text.clear(); l.toks.clear(); l.clear = true;   // (toks: the loop check reads this request's own tokens only)
                        l.head.clear();
                        for (size_t k = 0; k < l.req->prompt.size() && k < 40; ++k) { l.head += piece(l.req->prompt[k]); }
                        l.t_start = now_ms(); l.t_prompt = 0;
                        if (fused) {
                            llama_fq_set_sampling((int32_t) (&l - L.data()), l.req->temp, l.req->min_p, l.req->seed);
                            l.samp = l.req->temp > 0.0f;
                        }
                        if (l.req->fork_src >= 0) { forks.push_back({l.req->fork_src, l.idx}); }
                    }
                }
                // version 2: lanes that continue from a session get its memory, all lanes of one session in one call
                while (!forks.empty()) {
                    const int sl = forks.front().first;
                    std::vector<int32_t> dst;
                    for (auto it = forks.begin(); it != forks.end();) {
                        if (it->first == sl) { dst.push_back(it->second); it = forks.erase(it); } else { ++it; }
                    }
                    const bool ok = vault[sl].used && llama_fq_fork(ctx, lanes + sl, dst.data(), (int32_t) dst.size()) != 0;
                    for (int32_t i : dst) {
                        if (ok) { L[i].clear = false; L[i].pos = vault[sl].pos; L[i].own = 0; }
                        else    { L[i].failed = true; }
                    }
                    n_forks += (long long) dst.size();
                    vault[sl].refs -= (int) dst.size();
                    if (vault[sl].refs <= 0 && !vault[sl].user) { vault[sl].freeing = true; }
                }
                for (int i = 0; i < lanes; ++i) { if (L[i].req) { need = i + 1; } }
            }
            for (auto & l : L) {
                if (l.failed) { l.failed = false; l.text = "[the session could not be copied]"; l.t_prompt = now_ms(); finish(l, "error"); l.clear = true; }
            }

            // version 2: when the lanes still at work would fit a narrower pass, the upper ones move down (their memory is
            // copied on the card, a few milliseconds) and the pass narrows at once instead of dragging idle lanes along
            if (pack && need > 0) {
                int active = 0;
                for (auto & l : L) { active += l.req ? 1 : 0; }
                const int wt = width_for(active);
                bool queued;
                { std::lock_guard<std::mutex> lk(mu); queued = !queue.empty(); }
                if (wt < width_for(need) && wt < width && !queued && ++calm >= 6) {
                    int j = 0;
                    for (int i = lanes - 1; i >= wt; --i) {
                        if (!L[i].req) { continue; }
                        while (j < wt && L[j].req) { ++j; }
                        if (j >= wt) { break; }
                        const int32_t d = j;
                        if (!L[i].clear && llama_fq_fork(ctx, i, &d, 1) == 0) { break; }
                        Lane & a = L[i]; Lane & b = L[j];
                        b.req = std::move(a.req); a.req.reset();
                        b.n_fed = a.n_fed; b.n_gen = a.n_gen; b.pos = a.pos; b.last = a.last; b.clear = a.clear; b.expects = a.expects;
                        b.text = std::move(a.text); a.text.clear(); b.toks = std::move(a.toks); a.toks.clear(); b.head = a.head; b.t_start = a.t_start; b.t_prompt = a.t_prompt; b.own = a.own; b.fed_bytes = a.fed_bytes;
                        if (fused) {
                            llama_fq_set_sampling(j, b.req->temp, b.req->min_p, b.req->seed + (uint32_t) b.n_gen);
                            b.samp = b.req->temp > 0.0f;
                        }
                        llama_memory_seq_rm(mem, i, -1, -1);
                        a.pos = 0; a.own = 0; a.clear = true; a.fed_bytes = -1;
                        n_moves++;
                    }
                    need = 0;
                    for (int i = 0; i < lanes; ++i) { if (L[i].req) { need = i + 1; } }
                    if (width_for(need) < width) { width = width_for(need); narrow = 0; }
                    calm = 0;
                } else if (!(wt < width_for(need) && wt < width)) { calm = 0; }
            }

            // widen at once, narrow only after narrow_wait passes: a change of width costs about one extra pass (the graph is rebuilt)
            const int w = width_for(need);
            if (w > width) { width = w; narrow = 0; }
            // (but at once when the lanes at work fit a quarter of the width or less: after a long answer, the next short one
            // was written in passes as wide as the long one for 48 passes, 65 ms a token instead of 12 on Ternary-Bonsai-2-27B
            // with 156 lanes, 19 Nov 2026)
            else if (w < width) { if (++narrow >= (w * 4 <= width ? 3 : narrow_wait)) { width = w; narrow = 0; } }
            else { narrow = 0; }
            const int W = width;

            // one bulk pass for one of the lanes that read a long prompt, in turn
            n_busy = 0;
            for (int i = 0; i < W; ++i) { n_busy += L[i].req ? 1 : 0; }
            int reader = -1;
            for (int k = 0; k < W; ++k) {
                const int i = (rr + k) % W;
                if (bulk_left(L[i]) > 0) { reader = i; break; }
            }
            if (reader >= 0) {
                rr = reader + 1;
                if (!bulk_pass(batch, mem, reader, L[reader])) {
                    fprintf(stderr, "engine: bulk pass failed on lane %d\n", reader);
                    L[reader].t_prompt = now_ms(); finish(L[reader], "error"); L[reader].clear = true;
                } else if (L[reader].req->read_only && L[reader].n_fed >= L[reader].req->prompt.size()) {
                    L[reader].t_prompt = now_ms(); finish(L[reader], "read");      // version 2: read and parked
                }
                bool others = false;                       // a lane that generates, or reads its last tokens
                for (int i = 0; i < W; ++i) { others = others || (L[i].req && bulk_left(L[i]) == 0); }
                if (!others) { publish(); continue; }
            }

            const double t_pass = now_ms();

            batch.n_tokens = W;
            for (int i = 0; i < W; ++i) {
                Lane & l = L[i];
                llama_token tok;
                if (l.req) {
                    if (l.clear) { llama_memory_seq_rm(mem, i, -1, -1); l.pos = 0; l.own = 0; l.clear = false; }
                    const auto & p = l.req->prompt;
                    if (l.n_fed < p.size()) {
                        tok = p[l.n_fed++];
                        l.expects = l.n_fed == p.size() && !l.req->read_only;
                        n_prompt++;
                    } else {
                        tok = l.last;
                        l.expects = true;
                    }
                } else {
                    // an idle lane keeps the pass at its width with a filler token. Its sequence just grows and is emptied
                    // only when the lane's context is full: emptying means clearing the lane's state in every layer
                    // (about 3.5 ms), which until 18 Nov was done before every pass that had an idle lane
                    if (l.samp) { llama_fq_set_sampling(i, 0.0f, 0.0f, 0); l.samp = false; }
                    if (l.clear || l.own >= ctx_lane - 1) { llama_memory_seq_rm(mem, i, -1, -1); l.pos = 0; l.own = 0; }
                    l.clear   = false;
                    l.expects = false;
                    tok = filler;
                }
                l.own++;
                batch.token[i]     = tok;
                batch.pos[i]       = l.pos++;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = i;
                batch.logits[i]    = 1;
            }
            for (int i = W; i < lanes; ++i) { L[i].clear = true; }

            int rc;
            {
                std::lock_guard<std::mutex> pk(pass_mu);
                rc = llama_decode(ctx, batch);
            }
            if (rc != 0) {
                fprintf(stderr, "engine: llama_decode returned %d\n", rc);
                for (auto & l : L) {
                    if (l.req) { l.text += " [engine error]"; l.t_prompt = l.t_prompt ? l.t_prompt : now_ms(); finish(l, "error"); l.clear = true; }
                }
                continue;
            }

            for (int i = 0; i < W; ++i) {
                Lane & l = L[i];
                if (l.req && l.req->read_only && l.n_fed >= l.req->prompt.size()) { l.t_prompt = now_ms(); finish(l, "read"); continue; }
                if (!l.req || !l.expects) { continue; }
                if (l.req->cancelled) { l.t_prompt = l.t_prompt ? l.t_prompt : now_ms(); finish(l, "cancelled"); continue; }
                llama_token tok = llama_get_sampled_token_ith(ctx, i);
                if (tok == LLAMA_TOKEN_NULL) {
                    // sampling did not run on the card: arg max over the logits on the host (slow)
                    if (!warned) { fprintf(stderr, "engine: WARNING sampling runs on the host\n"); warned = true; }
                    const float * lg = llama_get_logits_ith(ctx, i);
                    const int nv = llama_vocab_n_tokens(vocab);
                    int best = 0;
                    for (int v = 1; v < nv; ++v) { if (lg[v] > lg[best]) { best = v; } }
                    tok = best;
                }
                if (l.n_gen == 0) { l.t_prompt = now_ms(); }
                if (llama_vocab_is_eog(vocab, tok)) { finish(l, "eos"); continue; }
                {
                    const std::string pc = piece(tok);
                    l.text += pc;
                    bool stopped = false;
                    for (const auto & st : l.req->stops) {          // only the end of the text can hold a new match
                        const size_t back = pc.size() + st.size();
                        const size_t at = l.text.find(st, l.text.size() > back ? l.text.size() - back : 0);
                        if (at != std::string::npos) { l.fed_bytes = (int) (l.text.size() - pc.size()); l.text.erase(at); stopped = true; break; }
                    }
                    if (stopped) { l.n_gen++; n_tokens++; finish(l, "stop"); continue; }
                    if (l.req->stream) { l.req->push(pc); }
                    if (l.req->soft > 0 && l.n_gen + 1 >= l.req->soft) {     // version 2: long enough - end with this sentence
                        const size_t e = l.text.find_last_not_of(" \n\r\t");
                        const size_t in_piece = l.text.size() - pc.size();
                        if (e != std::string::npos && e >= in_piece && (l.text[e] == '.' || l.text[e] == '!' || l.text[e] == '?')
                            && !(e > 0 && l.text[e] == '.' && isdigit((unsigned char) l.text[e - 1]))) {
                            l.n_gen++; n_tokens++; finish(l, "soft"); continue;
                        }
                    }
                }
                l.last  = tok;
                l.n_gen++;
                n_tokens++;
                if (l.req->loop > 0) {
                    // a lane caught in a loop (greedy writing does not get out of one: "The Eastern Empire was more stable /
                    // wealthy / powerful ... than the Western Empire.", 21 Nov): its last `loop` tokens stood in the same order
                    // earlier in what it wrote. On the host, between passes: no pass of the card
                    l.toks.push_back(tok);
                    const int K = l.req->loop, n = (int) l.toks.size();
                    if (n >= 2 * K) {
                        const llama_token * tail = l.toks.data() + n - K;
                        for (int a = 0; a + K <= n - K; ++a) {
                            if (l.toks[a] == tail[0] && std::equal(tail, tail + K, l.toks.data() + a)) { finish(l, "loop"); break; }
                        }
                        if (!l.req) { continue; }
                    }
                }
                if (l.n_gen >= l.req->max_tokens) { finish(l, "limit"); }
            }
            n_pass++;
            ms_pass = ms_pass + (now_ms() - t_pass);
            publish();
        }
        llama_batch_free(batch);
    }
};

static std::string chatml(const json & messages, bool thinking) {
    std::string s;
    for (const auto & m : messages) {
        std::string content;
        if (m.contains("content") && m["content"].is_string()) {
            content = m["content"].get<std::string>();
        } else if (m.contains("content") && m["content"].is_array()) {
            for (const auto & part : m["content"]) {
                if (part.value("type", "") == "text") { content += part.value("text", ""); }
            }
        }
        s += "<|im_start|>" + m.value("role", "user") + "\n" + content + "<|im_end|>\n";
    }
    s += "<|im_start|>assistant\n";
    s += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    return s;
}

static std::string g_model_id = "local";   // the loaded model's file name, for the OpenAI-style answers

int main(int argc, char ** argv) {
    std::string model_path;
    int lanes = 64, ctx_lane = 384, port = 11832, threads = 4, trickle = 96, kvu = 1, top_k = 40, state_bits = 16, chunk = 512;
    bool alone_bulk = true;
    int few_bulk = 8;
    int narrow_wait = getenv("RAPIDLI_NARROW_WAIT") ? std::max(1, atoi(getenv("RAPIDLI_NARROW_WAIT"))) : 8;
    float temp = 0.0f, top_p = 0.95f, min_p = 0.05f;
    unsigned seed = 1;
    int use_chains = 0;
    int n_vault = 0;          // version 2: sessions (parked states) that can exist at the same time
    int n_pack  = 1;
    int n_spec  = 0;
    int ctx_extra = 0;         // version 2: attention cells reserved for long parked prompts (a chat's history)
    std::string mmproj_path;   // version 2: the model's vision projector (images)
    int spec_batch = 256;      // tokens read per pass by /spec (19 Nov: 256, was 64: cold reads ran ~3x slower than the lanes)
    int spec_ctx = 4096;       // version 2: tokens of context of the drafting pair
    int n_lookup = 0;          // version 3: tokens drafted by copying from the text (0 = off); /spec runs on it alone without an MTP head
    std::string kv_type = "f16";   // version 2: type of the lanes' attention cache (f16 or q8_0)           // version 2: tokens drafted per cycle by the MTP head on /spec (0 = no /spec)
    // version 3: models larger than the card. --experts-ram 1 keeps the routed experts of a mixture-of-experts model in
    // the PC's memory (read from the file as it is mapped, no copy); everything else stays on the card. The card keeps
    // the experts used most in a cache of --moe-cache-mib MiB (it follows what is used), and --moe-cpu-share of the
    // experts it misses are computed by the processor where they are instead of being copied to the card.
    int experts_ram = 0;
    long long moe_cache_mib = 0;
    long long moe_spec_mib = 0;      // the cache of /spec's context (the drafted single answer)
    float moe_cpu_share = 0.0f;
    float moe_spec_share = -1.0f;    // the share of /spec's cache misses on the processor (-1: as --moe-cpu-share)
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if      (a == "--model")   { model_path = argv[i + 1]; }
        else if (a == "--lanes")   { lanes      = atoi(argv[i + 1]); }
        else if (a == "--ctx")     { ctx_lane   = atoi(argv[i + 1]); }
        else if (a == "--port")    { port       = atoi(argv[i + 1]); }
        else if (a == "--threads") { threads    = atoi(argv[i + 1]); }
        else if (a == "--trickle") { trickle    = atoi(argv[i + 1]); }
        else if (a == "--alone-bulk") { alone_bulk = atoi(argv[i + 1]) != 0; }
        else if (a == "--few-bulk") { few_bulk = std::max(1, atoi(argv[i + 1])); }
        else if (a == "--narrow-wait") { narrow_wait = std::max(1, atoi(argv[i + 1])); }
        else if (a == "--kvu")     { kvu        = atoi(argv[i + 1]); }
        else if (a == "--chunk")   { chunk      = std::max(8, atoi(argv[i + 1])); }
        else if (a == "--temp")    { temp       = (float) atof(argv[i + 1]); }
        else if (a == "--min-p")   { min_p      = (float) atof(argv[i + 1]); }
        else if (a == "--chains")  { use_chains = atoi(argv[i + 1]); }
        else if (a == "--top-k")   { top_k      = atoi(argv[i + 1]); }
        else if (a == "--top-p")   { top_p      = (float) atof(argv[i + 1]); }
        else if (a == "--seed")    { seed       = (unsigned) atoi(argv[i + 1]); }
        else if (a == "--state")   { state_bits = atoi(argv[i + 1]); }
        else if (a == "--vault")   { n_vault    = std::max(0, atoi(argv[i + 1])); }
        else if (a == "--pack")    { n_pack     = atoi(argv[i + 1]); }
        else if (a == "--spec")    { n_spec     = std::max(0, atoi(argv[i + 1])); }
        else if (a == "--spec-ctx") { spec_ctx  = std::max(512, atoi(argv[i + 1])); }
        else if (a == "--lookup")  { n_lookup   = std::max(0, std::min(16, atoi(argv[i + 1]))); }
        else if (a == "--spec-batch") { spec_batch = std::max(16, std::min(1024, atoi(argv[i + 1]))); }
        else if (a == "--ctx-extra") { ctx_extra = std::max(0, atoi(argv[i + 1])); }
        else if (a == "--mmproj")  { mmproj_path = argv[i + 1]; }
        else if (a == "--kv")      { kv_type    = argv[i + 1]; }
        else if (a == "--experts-ram")   { experts_ram   = atoi(argv[i + 1]); }
        else if (a == "--moe-cache-mib") { moe_cache_mib = std::max(0LL, atoll(argv[i + 1])); }
        else if (a == "--moe-spec-mib")  { moe_spec_mib  = std::max(0LL, atoll(argv[i + 1])); }
        else if (a == "--moe-cpu-share") { moe_cpu_share = std::min(1.0f, std::max(0.0f, (float) atof(argv[i + 1]))); }
        else if (a == "--moe-spec-cpu-share") { moe_spec_share = std::min(1.0f, std::max(0.0f, (float) atof(argv[i + 1]))); }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty()) { fprintf(stderr, "usage: rapidli-engine --model <gguf> [--lanes 64] [--ctx 384] [--port 11832] [--threads 4] [--trickle 96] [--chunk 512] [--kvu 1] [--state 16|8] [--temp 0] [--min-p 0.05] [--chains 0 [--top-k 40] [--top-p 0.95] [--seed 1]] [--vault 0]\n"); return 2; }
    {
        std::string b = model_path.substr(model_path.find_last_of("/\\") + 1);
        if (b.size() > 5 && b.compare(b.size() - 5, 5, ".gguf") == 0) { b.resize(b.size() - 5); }
        if (!b.empty()) { g_model_id = b; }
    }

    // switches of our llama.dll / ggml-cuda.dll (see engine/llama/patch_*.py); values already in the environment win
    auto env_default = [](const char * name, const char * value) { 
#if defined(_WIN32)
        if (getenv(name) == nullptr) { _putenv_s(name, value); }
#else
        setenv(name, value, 0);                 // (0: a value already set stays)
#endif
    };
    env_default("LLAMA_RS_DIRECT", "1");                    // the recurrent state is updated in place
    env_default("GGML_CUDA_LAZY_MEMSET", "1");              // clearing a starting lane's state: one wait for the card, not 96
    env_default("LLAMA_RS_DIRECT_MAX", std::to_string(chunk).c_str());   // ... also in the bulk passes
    env_default("LLAMA_KV_PAD", "1000000");                 // the attention cache is always seen whole: the graph never changes
    if (state_bits == 8) { env_default("LLAMA_RS_I8", "1"); } else { env_default("LLAMA_RS_F16", "1"); }
    if (!use_chains) { env_default("LLAMA_FAST_GREEDY", "1"); }
    if (n_spec > 0 || n_lookup > 0) { env_default("LLAMA_RS_SPEC_F16", "1"); }    // the drafting context's state in 16 bits: half the memory

    ggml_backend_load_all();
    llama_backend_init();

    Engine eng;
    eng.lanes = lanes; eng.ctx_lane = ctx_lane; eng.ctx_extra = ctx_extra; eng.trickle = trickle; eng.chunk = chunk; eng.alone_bulk = alone_bulk; eng.few_bulk = few_bulk; eng.narrow_wait = narrow_wait;
    eng.fused = !use_chains; eng.def_temp = temp; eng.def_min_p = min_p;
    eng.vault.resize(n_vault);
    eng.pack = n_pack;
    // a pass costs about 19.5 ms + 0.4 ms per lane whether the lane is in use or not, so the steps are small
    for (int w : {1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 144}) { if (w < lanes) { eng.widths.push_back(w); } }
    eng.widths.push_back(lanes);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mtp = n_spec > 0;
    std::vector<llama_model_tensor_buft_override> overrides;
    if (experts_ram) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) { fprintf(stderr, "engine: no CPU backend for the experts\n"); return 1; }
        // the routed experts (not the shared ones, which every token uses): gate, up, down, or gate and up fused
        overrides.push_back({ "\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_dev_buffer_type(cpu) });
        overrides.push_back({ nullptr, nullptr });
        mp.tensor_buft_overrides = overrides.data();
        fprintf(stderr, "engine: experts in the PC's memory, a cache of %lld MiB on the card, %.2f of its misses on the processor\n",
                moe_cache_mib, moe_cpu_share);
    }
    eng.model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!eng.model) { fprintf(stderr, "engine: the model did not load\n"); return 1; }
    if (n_spec > 0 && llama_model_n_layer_nextn(eng.model) <= 0) {
        fprintf(stderr, n_lookup > 0 ? "engine: the model has no multi-token-prediction head: /spec drafts by copying from the text only\n"
                                     : "engine: the model has no multi-token-prediction head: no drafting (/spec off)\n");
        n_spec = 0;
    }
    eng.vocab = llama_model_get_vocab(eng.model);

    std::vector<llama_sampler_seq_config> samplers(lanes);
    for (int i = 0; i < lanes; ++i) {
        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (!use_chains || temp <= 0.0f) {
            // greedy. With LLAMA_FAST_GREEDY=1 the library replaces these chains by one operation over all lanes; without
            // it, temperature 0 first reduces the row of logits to its largest entry on the card, so the engine does not
            // copy the whole row (152k numbers per lane per pass) to the host; a bare "greedy" keeps the row as an output
            llama_sampler_chain_add(chain, llama_sampler_init_temp(0.0f));
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        } else {
            // --chains 1: llama.cpp's own samplers, the same for every request (the chains are part of the recorded
            // graph): a chain per lane on the card, about 15% slower at 80 lanes
            if (top_k > 0)    { llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k)); }
            if (top_p < 1.0f) { llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1)); }
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
            llama_sampler_chain_add(chain, llama_sampler_init_dist(seed + (unsigned) i));
        }
        samplers[i] = { i, chain };
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                 = (uint32_t) ((lanes + n_vault) * ctx_lane + ctx_extra);
    cp.n_batch               = std::max(lanes, eng.chunk);
    cp.n_ubatch              = std::max(lanes, eng.chunk);
    cp.n_seq_max             = lanes + n_vault;
    cp.n_outputs_max         = lanes + n_vault;
    cp.n_outputs_max_per_seq = 1;
    cp.n_threads             = threads;
    cp.n_threads_batch       = threads;
    // one attention cache per lane (default): a lane attends over its own cells only. --kvu 1: one cache shared by all
    // lanes, where every lane scans the cells of all lanes (a mask of lanes x all cells is built and sent in every pass)
    cp.kv_unified            = kvu != 0;
    const ggml_type kvt = kv_type == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    if (kvt != GGML_TYPE_F16) {
        cp.type_k = kvt; cp.type_v = kvt;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;     // a quantised V cache needs flash attention
        fprintf(stderr, "engine: attention cache %s\n", kv_type.c_str());
    }
    cp.no_perf               = true;
    if (experts_ram && moe_cache_mib > 0) {
        cp.moe_cache_size      = (size_t) moe_cache_mib * 1024 * 1024;
        cp.moe_cache_cpu_share = moe_cpu_share;
    }
    cp.samplers              = samplers.data();
    cp.n_samplers            = samplers.size();
    eng.ctx = llama_init_from_model(eng.model, cp);
    if (!eng.ctx) { fprintf(stderr, "engine: the context did not initialise\n"); return 1; }

    {
        const auto t = eng.tokenize(" the");
        eng.filler = t.empty() ? 0 : t.back();
    }

    Spec spec;
    if (n_spec > 0 || n_lookup > 0) {
        spec.n_draft  = n_spec;
        spec.n_lookup = n_lookup;
        spec.n_batch  = std::max(spec_batch, 1 + std::max(n_spec, n_lookup));
        spec.n_embd   = llama_model_n_embd(eng.model);
        spec.n_vocab  = llama_vocab_n_tokens(eng.vocab);
        llama_context_params sp = llama_context_default_params();
        // the full model can take back as many tokens as the longest draft (a snapshot of its recurrent state per token)
        sp.n_ctx = (uint32_t) spec_ctx; sp.n_batch = spec.n_batch; sp.n_ubatch = spec.n_batch; sp.n_seq_max = 1; sp.n_rs_seq = (uint32_t) std::max(n_spec, n_lookup);
        sp.n_threads = threads; sp.n_threads_batch = threads; sp.no_perf = true;
        if (kvt != GGML_TYPE_F16) { sp.type_k = kvt; sp.type_v = kvt; sp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; }
        // (each context has a cache of its own: the one answer /spec writes token by token gains most from it)
        // (one answer, token by token, gains from every miss copied to the card; the lanes' wide passes from the
        // processor taking part of them: 45.3 against 32.4 tok/s, and 104 against 130 tok/s with 16 lanes, 20 Nov)
        if (experts_ram && moe_spec_mib > 0) { sp.moe_cache_size = (size_t) moe_spec_mib * 1024 * 1024; sp.moe_cache_cpu_share = moe_spec_share >= 0 ? moe_spec_share : moe_cpu_share; }
        spec.tgt = llama_init_from_model(eng.model, sp);
        if (!spec.tgt) { fprintf(stderr, "engine: the /spec context did not initialise\n"); return 1; }
        if (n_spec > 0) {
            llama_context_params dp = llama_context_default_params();
            dp.n_ctx = (uint32_t) spec_ctx; dp.n_batch = spec.n_batch; dp.n_ubatch = spec.n_batch; dp.n_seq_max = 1; dp.n_rs_seq = 0;
            dp.ctx_type = LLAMA_CONTEXT_TYPE_MTP; dp.ctx_other = spec.tgt; dp.n_threads = threads; dp.n_threads_batch = threads; dp.no_perf = true;
            spec.dft = llama_init_from_model(eng.model, dp);
            if (!spec.dft) { fprintf(stderr, "engine: the /spec contexts did not initialise\n"); return 1; }
            llama_set_embeddings_nextn(spec.tgt, true, false);
            llama_set_embeddings_nextn(spec.dft, true, true);
            spec.db = llama_batch_ext_init(spec.dft);
        }
        spec.tb = llama_batch_init(spec.n_batch, 0, 1);
        // the lookup's prior: a copy that matched more tokens is kept more often (corrected by what is measured)
        for (int n = 1; n <= 8; ++n) { spec.lk_acc[n] = std::min(0.85, 0.3 + 0.1 * n); }
        for (int k = 1; k <= 16; ++k) { spec.mtp_acc[k] = 0.8; }
        fprintf(stderr, "engine: /spec ready, %d tokens drafted per cycle by the MTP head, up to %d copied from the text\n", n_spec, n_lookup);
    }

    if (!mmproj_path.empty()) {
        mtmd_context_params mp2 = mtmd_context_params_default();
        mp2.use_gpu = true; mp2.n_threads = threads; mp2.print_timings = false; mp2.warmup = false;
        eng.mctx = mtmd_init_from_file(mmproj_path.c_str(), eng.model, mp2);
        if (!eng.mctx) { fprintf(stderr, "engine: the vision projector did not load: %s\n", mmproj_path.c_str()); }
        else { fprintf(stderr, "engine: vision projector loaded (images)\n"); }
    }

    std::thread worker([&] { eng.run(); });

    // version 4: the engine's own speed (22 Nov). A thread notes, on the engine's clock, each moment the count of written
    // tokens grows (the end of a pass, a drafted cycle); /lanes and /stats report the last second's tokens a second and
    // /speed an answer's best second. The app's meter, an answer's peak and the Top 10 read this one measurement instead
    // of each timing the counter from outside (one answer showed 500 live, 510 under it and 501 in its saved figures).
    struct SpeedLog { std::mutex mu; std::deque<std::pair<double, long long>> marks; long long last = 0; } speed;
    std::atomic<bool> speed_stop{false};
    auto total_tokens = [&]() -> long long {
        long long t = (long long) eng.n_tokens + (long long) spec.n_tokens;
        if (spec.tgt) { std::lock_guard<std::mutex> vk(spec.view_mu); if (spec.active) { t += spec.v_tokens; } }
        return t;
    };
    std::thread speed_th([&] {
        while (!speed_stop) {
            const long long t = total_tokens();
            {
                std::lock_guard<std::mutex> lk(speed.mu);
                // (never down: a drafted answer's tokens leave its live count a moment before they join the total)
                if (speed.marks.empty() || t > speed.last) {
                    speed.last = std::max(t, speed.last);
                    speed.marks.emplace_back(now_ms(), speed.last);
                    while (speed.marks.size() > 40000) { speed.marks.pop_front(); }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }
    });
    // tokens a second over the last second: whole passes while the engine writes (from a mark a second or more before the
    // last one to the last one), and down to 0 once it stops (the window then ends now)
    auto rate_now = [&](double now) -> double {
        std::lock_guard<std::mutex> lk(speed.mu);
        if (speed.marks.size() < 2) { return 0.0; }
        const auto & b = speed.marks.back();
        const double end = now - b.first < 500.0 ? b.first : now;
        for (auto it = speed.marks.rbegin(); it != speed.marks.rend(); ++it) {
            if (end - it->first >= 1000.0) { return (b.second - it->second) * 1000.0 / (end - it->first); }
        }
        const auto & f = speed.marks.front();
        return end - f.first >= 200.0 ? (b.second - f.second) * 1000.0 / (end - f.first) : 0.0;
    };
    // the best second between two moments of the engine's clock (the tokens of every answer written meanwhile)
    auto rate_peak = [&](double from, double to) -> double {
        std::lock_guard<std::mutex> lk(speed.mu);
        std::vector<std::pair<double, long long>> m;
        for (const auto & x : speed.marks) { if (x.first >= from - 1000.0 && x.first <= to + 100.0) { m.push_back(x); } }
        double best = 0.0;
        size_t j = 0;
        for (size_t i = 0; i < m.size(); ++i) {
            if (m[i].first < from) { continue; }
            while (j + 1 < i && m[i].first - m[j + 1].first >= 1000.0) { ++j; }
            const double dt = m[i].first - m[j].first;
            if (dt >= 1000.0) { best = std::max(best, (m[i].second - m[j].second) * 1000.0 / dt); }
        }
        return best;
    };

    httplib::Server svr;
    svr.set_pre_routing_handler([](const httplib::Request & req, httplib::Response & res) {
        const std::string host = req.get_header_value("Host");
        const bool own = host.empty() || host.rfind("127.0.0.1:", 0) == 0 || host.rfind("localhost:", 0) == 0 || host.rfind("[::1]:", 0) == 0;
        if (!req.has_header("Origin") && own) { return httplib::Server::HandlerResponse::Unhandled; }
        res.status = 403;
        res.set_content("{\"error\":\"not allowed\"}", "application/json");
        return httplib::Server::HandlerResponse::Handled;
    });
    svr.new_task_queue = [] { return new httplib::ThreadPool(512); };

    // puts a request into the queue; nullptr (and err) when it cannot be served
    auto number = [](const json & body, const char * key, double def) {
        return body.contains(key) && body[key].is_number() ? body[key].get<double>() : def;
    };
    auto enqueue = [&](std::vector<llama_token> toks, int max_tokens, bool stream, std::string & err, const json & body,
                       std::shared_ptr<Group> group = nullptr, int branch = 0) -> std::shared_ptr<Request> {
        g_warmer.cancel();                      // real work: the warm-up stops at its next multiplication
        auto r = std::make_shared<Request>();
        r->read_only = body.value("read_only", false);
        r->hold      = body.value("hold", false);
        r->bulk      = body.value("bulk", false);
        r->fork_src  = body.contains("session") && body["session"].is_number_integer() ? body["session"].get<int>() : -1;
        r->group     = group;
        r->branch    = branch;
        r->soft      = body.value("soft", 0);
        r->loop      = body.value("loop", 0);
        r->name      = body.contains("name") && body["name"].is_string() ? body["name"].get<std::string>() : std::string();
        r->race      = body.contains("race") && body["race"].is_number_integer() ? body["race"].get<int>() : -1;
        if (body.contains("stop") && body["stop"].is_array()) {
            for (const auto & st : body["stop"]) { if (st.is_string() && !st.get<std::string>().empty()) { r->stops.push_back(st.get<std::string>()); } }
        }
        if (r->fork_src >= 0) {
            std::lock_guard<std::mutex> lk(eng.mu);
            if (r->fork_src >= (int) eng.vault.size() || !eng.vault[r->fork_src].used || eng.vault[r->fork_src].freeing) { err = "no such session"; return nullptr; }
            if (eng.vault[r->fork_src].pending >= 0) { toks.insert(toks.begin(), eng.vault[r->fork_src].pending); }
            if (toks.empty()) { err = "nothing to read"; return nullptr; }
            if ((int) toks.size() + 1 >= (r->read_only ? std::max(8 * eng.ctx_lane, eng.ctx_extra) : 1 * eng.ctx_lane)) { err = "the prompt does not fit the lane context"; return nullptr; }
            eng.vault[r->fork_src].refs++;
        }
        if (toks.empty()) { err = "nothing to read"; return nullptr; }
        if ((int) toks.size() + 1 >= (r->read_only ? std::max(8 * eng.ctx_lane, eng.ctx_extra) : 1 * eng.ctx_lane)) { err = "the prompt does not fit the lane context"; return nullptr; }
        r->max_tokens = std::max(1, std::min(max_tokens, eng.ctx_lane - (int) toks.size() - 1));
        r->prompt     = std::move(toks);
        r->t_arrive   = now_ms();
        r->stream     = stream;
        r->temp       = (float) std::max(0.0, number(body, "temperature", eng.def_temp));
        r->min_p      = (float) std::min(1.0, std::max(0.0, number(body, "min_p", eng.def_min_p)));
        r->seed       = body.contains("seed") && body["seed"].is_number_integer() ? (uint32_t) body["seed"].get<long long>() : eng.seed_next++ * 2654435761u;
        // the prompt takes part in the seed: the same prompt with the same seed repeats its answer, but different
        // prompts sent with one seed do not share their random numbers (measured: shared numbers cost 5 of 300
        // answers at temperature 0.7, because the unlucky draws then hit every request at once)
        for (llama_token t : r->prompt) { r->seed = (r->seed ^ (uint32_t) t) * 16777619u; }
        {
            std::lock_guard<std::mutex> lk(eng.mu);
            eng.queue.push_back(r);
        }
        eng.cv.notify_one();
        return r;
    };
    auto submit = [&](std::vector<llama_token> toks, int max_tokens, json & out, std::string & err, const json & body) {
        auto r = enqueue(std::move(toks), max_tokens, false, err, body);
        if (!r) { return false; }
        out = r->done.get_future().get();
        return true;
    };

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"object", "list"}, {"data", json::array({json{{"id", g_model_id}, {"object", "model"}, {"owned_by", "local"}}})}}.dump(), "application/json");
    });
    svr.Get("/stats", [&](const httplib::Request &, httplib::Response & res) {
        const long long np = eng.n_pass;
        json s = {{"lanes", eng.lanes}, {"width", (int) eng.width}, {"prefill_passes", (long long) eng.n_prefill}, {"passes", np}, {"mean_pass_ms", np ? eng.ms_pass / np : 0.0}, {"generated_tokens", (long long) eng.n_tokens},
                  {"prompt_tokens", (long long) eng.n_prompt}, {"requests_done", (long long) eng.n_done},
                  {"sessions", (int) eng.n_sessions}, {"session_slots", (int) eng.vault.size()}, {"forks", (long long) eng.n_forks}, {"moves", (long long) eng.n_moves},
                  {"model", model_path}, {"vision", eng.mctx != nullptr}, {"ctx_lane", eng.ctx_lane}, {"ctx_extra", eng.ctx_extra}, {"spec_ctx", spec.tgt ? spec_ctx : 0}, {"spec_batch", spec.n_batch}, {"spec_slots", 4},
                  {"vault", (int) eng.vault.size()},
                  // "spec": /spec exists (tokens drafted per cycle at most); "spec_mtp" / "spec_lookup": by which drafter
                  {"spec", spec.tgt ? std::max(spec.n_draft, spec.n_lookup) : 0}, {"spec_mtp", spec.n_draft}, {"spec_lookup", spec.n_lookup},
                  {"spec_answers", (long long) spec.n_answers}, {"spec_tokens", (long long) spec.n_tokens},
                  {"spec_kept", (long long) spec.n_kept}, {"spec_drafted", (long long) spec.n_drafted}, {"spec_prefix_hits", (long long) spec.n_px_hits},
                  {"lookup_cycles", (long long) spec.n_lk_cycles}, {"lookup_kept", (long long) spec.n_lk_kept}, {"lookup_drafted", (long long) spec.n_lk_drafted},
                  // version 3: the routed experts are in the PC's memory (each context with its cache on the card)
                  {"experts_ram", experts_ram != 0}, {"moe_cache_mib", moe_cache_mib}, {"moe_spec_mib", moe_spec_mib}};
        const double t_now = now_ms();
        s["clock_ms"] = t_now; s["rate"] = rate_now(t_now); s["total_tokens"] = total_tokens();   // version 4
        res.set_content(s.dump(), "application/json");
    });
    // ---- version 2 ----
    // POST /spec {"prompt": "full text to read", "n_predict": n, "stop": [...]}: one answer, drafted by the MTP head and
    // verified by the full model. Streams {"rows": [[0, "text"]]} and ends with {"done": [{...}], "timings": {...}} like /fork.
    svr.Post("/spec", [&](const httplib::Request & req, httplib::Response & res) {
        json body = json::parse(req.body, nullptr, false);
        if (!spec.tgt || body.is_discarded() || !body.contains("prompt") || !body["prompt"].is_string()
            || (body.contains("n_predict") && !body["n_predict"].is_number_integer())) {
            res.status = 400; res.set_content("{\"error\":\"no /spec (start with --spec N) or no prompt\"}", "application/json"); return;
        }
        if (!spec.busy.try_lock()) { res.status = 409; res.set_content("{\"error\":\"/spec is busy\"}", "application/json"); return; }
        auto prompt = std::make_shared<std::vector<llama_token>>(eng.tokenize(body["prompt"].get<std::string>()));
        const std::string prefix = body.contains("prefix") && body["prefix"].is_string() ? body["prefix"].get<std::string>() : std::string();
        const int n_predict = body.value("n_predict", 400);
        std::vector<std::string> stops;
        if (body.contains("stop") && body["stop"].is_array()) { for (auto & x : body["stop"]) { if (x.is_string() && !x.get<std::string>().empty()) { stops.push_back(x.get<std::string>()); } } }
        res.set_header("Cache-Control", "no-store");
        res.set_chunked_content_provider("text/event-stream",
            [&, prompt, n_predict, stops, prefix](size_t, httplib::DataSink & sink) {
                std::unique_lock<std::mutex> own(spec.busy, std::adopt_lock);
                const double t0 = now_ms();
                const int E = spec.n_embd, V = spec.n_vocab, NB = spec.n_batch;
                const bool mtp = spec.dft != nullptr;      // without an MTP head: drafts copied from the text only
                llama_memory_t mt = llama_get_memory(spec.tgt), md = mtp ? llama_get_memory(spec.dft) : nullptr;
                llama_memory_clear(mt, true);
                if (mtp) { llama_memory_clear(md, true); }
                std::vector<float> pending(E, 0.0f), hrows, h(E);
                auto & tb = spec.tb;
                bool ok = true;
                // the MTP head reads the pairs (token at p, h of p-1) of what the full model has just read
                auto catch_up = [&](const llama_token * toks, int n, llama_pos pos0) -> bool {
                    if (!mtp) { return true; }
                    llama_batch_ext_clear(spec.db);
                    for (int k = 0; k < n; ++k) {
                        const int idx = llama_batch_ext_add_token(spec.db, 0, toks[k]);
                        const llama_pos p = pos0 + k;
                        llama_batch_ext_set_pos(spec.db, idx, &p);
                        const float * hp = k == 0 ? pending.data() : hrows.data() + (size_t) (k - 1) * E;
                        llama_batch_ext_set_embd_state(spec.db, idx, { hp, 1, (size_t) E });
                        llama_batch_ext_set_output_logits(spec.db, idx, false);
                    }
                    std::lock_guard<std::mutex> pk(eng.pass_mu);
                    return llama_process(spec.dft, LLAMA_PROCESS_TYPE_DECODE, spec.db) == 0;
                };
                // the full model reads toks[0..n) at pos0.., keeps h of every row in hrows; logits of the last row only
                // unless all_logits
                auto read = [&](const llama_token * toks, int n, llama_pos pos0, bool all_logits) -> bool {
                    for (int k = 0; k < n; ++k) { tb.token[k] = toks[k]; tb.pos[k] = pos0 + k; tb.n_seq_id[k] = 1; tb.seq_id[k][0] = 0; tb.logits[k] = all_logits || k == n - 1; }
                    tb.n_tokens = n;
                    std::lock_guard<std::mutex> pk(eng.pass_mu);
                    const double tr = now_ms();
                    if (llama_decode(spec.tgt, tb) != 0) { return false; }
                    if (all_logits && n <= 32) {               // a check of n rows: what such a pass costs (the choice of drafts)
                        llama_synchronize(spec.tgt);
                        const double ms = now_ms() - tr;
                        spec.pass_ms[n] = spec.pass_ms[n] > 0 ? spec.pass_ms[n] * 0.9 + ms * 0.1 : ms;
                    }
                    if (mtp) {
                        hrows.resize((size_t) n * E);
                        for (int k = 0; k < n; ++k) { std::memcpy(hrows.data() + (size_t) k * E, llama_get_embeddings_nextn_ith(spec.tgt, k), E * sizeof(float)); }
                    }
                    return true;
                };
                // 0. the parked prefix: restored, or read once and parked
                int start = 0;
                if (!prefix.empty()) {
                    Spec::Prefix * hit = nullptr;
                    for (auto & x : spec.px) { if (x.text == prefix && !x.tgt.empty()) { hit = &x; } }
                    if (hit) {
                        std::lock_guard<std::mutex> pk(eng.pass_mu);
                        ok = llama_state_seq_set_data(spec.tgt, hit->tgt.data(), hit->tgt.size(), 0) > 0 &&
                             (!mtp || llama_state_seq_set_data(spec.dft, hit->dft.data(), hit->dft.size(), 0) > 0);
                        if (ok) { pending = hit->pend; start = hit->len; spec.n_px_hits++; hit->used = ++spec.px_clock; }
                        else { llama_memory_clear(mt, true); if (mtp) { llama_memory_clear(md, true); } ok = true; }
                    }
                    if (start == 0) {
                        const std::vector<llama_token> xt = eng.tokenize(prefix);
                        for (int i = 0; ok && i < (int) xt.size(); i += NB) {
                            const int n = std::min(NB, (int) xt.size() - i);
                            ok = read(xt.data() + i, n, i, false) && catch_up(xt.data() + i, n, i);
                            if (ok && mtp) { std::memcpy(pending.data(), hrows.data() + (size_t) (n - 1) * E, E * sizeof(float)); }
                        }
                        if (ok) {
                            std::lock_guard<std::mutex> pk(eng.pass_mu);
                            Spec::Prefix * lru = &spec.px[0];
                            for (auto & y : spec.px) { if (y.used < lru->used) { lru = &y; } }
                            Spec::Prefix & x = *lru;
                            x.tgt.resize(llama_state_seq_get_size(spec.tgt, 0));
                            x.dft.resize(mtp ? llama_state_seq_get_size(spec.dft, 0) : 0);
                            const bool saved = llama_state_seq_get_data(spec.tgt, x.tgt.data(), x.tgt.size(), 0) > 0 &&
                                               (!mtp || llama_state_seq_get_data(spec.dft, x.dft.data(), x.dft.size(), 0) > 0);
                            x.text = saved ? prefix : std::string();
                            x.pend = pending;
                            x.len  = (int) xt.size();
                            x.used = ++spec.px_clock;
                            start = (int) xt.size();
                        }
                    }
                }
                // 1. the prompt, in chunks
                const auto & pt = *prompt;
                llama_token last = -1;
                for (int i = 0; ok && i < (int) pt.size(); i += NB) {
                    const int n = std::min(NB, (int) pt.size() - i);
                    ok = read(pt.data() + i, n, start + i, false) && catch_up(pt.data() + i, n, start + i);
                    if (ok) {
                        if (mtp) { std::memcpy(pending.data(), hrows.data() + (size_t) (n - 1) * E, E * sizeof(float)); }
                        if (i + n == (int) pt.size()) { last = argmax_row(llama_get_logits_ith(spec.tgt, n - 1), V); }
                    }
                }
                // the text so far, for drafts copied from it: the parked prefix, the prompt, then what is written
                std::vector<llama_token> hist;
                if (spec.n_lookup > 0) {
                    if (!prefix.empty()) { hist = eng.tokenize(prefix); }
                    hist.insert(hist.end(), pt.begin(), pt.end());
                }
                // the longest end of hist + [last] (2 to 8 tokens) that occurred before, the latest such place: the
                // tokens that followed it there are the draft. -> the length matched (0: none) and the draft in out
                auto look_up = [&](llama_token cur, int most, std::vector<llama_token> & out) -> int {
                    out.clear();
                    const int N = (int) hist.size() + 1;          // the sequence is hist followed by cur
                    auto at = [&](int i) { return i == N - 1 ? cur : hist[i]; };
                    if (most <= 0 || N < 3) { return 0; }
                    for (int n = std::min(8, N - 1); n >= 2; --n) {
                        for (int p = N - n - 1; p >= 0; --p) {     // a match at p .. p+n-1, followed by p+n (< N)
                            int m = 0;
                            while (m < n && at(p + m) == at(N - n + m)) { ++m; }
                            if (m < n) { continue; }
                            for (int q = p + n; q < N && (int) out.size() < most; ++q) { out.push_back(at(q)); }
                            if (!out.empty()) { return n; }
                        }
                    }
                    return 0;
                };
                // what a check of r rows costs: measured, or estimated from what was measured (a row more costs little)
                auto cost = [&](int r) -> double {
                    if (r <= 32 && spec.pass_ms[r] > 0) { return spec.pass_ms[r]; }
                    const double one = spec.pass_ms[1] > 0 ? spec.pass_ms[1] : 12.0;
                    return one * (1.0 + 0.05 * (r - 1));
                };
                const double t_read = now_ms();
                std::string text, why = ok ? "limit" : "error";
                {
                    std::lock_guard<std::mutex> vk(spec.view_mu);
                    spec.active = true; spec.v_tokens = 0; spec.v_kept = 0; spec.v_drafted = 0; spec.v_tail.clear(); spec.v_head.clear();
                    for (size_t k = 0; k < pt.size() && k < 40; ++k) { spec.v_head += eng.piece(pt[k]); }
                }
                int n_out = 0, cycles = 0, kept = 0, drafted = 0;
                llama_pos pos0 = (llama_pos) (start + pt.size());
                auto emit = [&](const std::string & piece) -> bool {
                    if (piece.empty()) { return true; }
                    const std::string line = "data: " + json({{"rows", json::array({json::array({0, piece})})}}).dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                    return sink.write(line.data(), line.size());
                };
                while (ok && n_out < n_predict) {
                    if (llama_vocab_is_eog(eng.vocab, last)) { why = "eos"; break; }
                    // 2. draft: copied from the text, by the MTP head, or none, whichever is expected to give the most
                    // tokens a millisecond: (1 + tokens expected to be kept) / (the check's cost + the drafting's), from
                    // the costs and keep rates measured so far. The check's cost is measured for every number of rows,
                    // and the drafts are cut to the length that pays.
                    std::vector<llama_token> d, lk;
                    const int room = n_predict - n_out - 1;
                    const int n_lk = room > 0 ? look_up(last, std::min(spec.n_lookup, room), lk) : 0;
                    double best = 1.0 / cost(1);
                    int use_mtp = 0, use_lk = 0;
                    if (mtp && room > 0) {
                        double e = 0, pr = 1;
                        for (int k = 1; k <= std::min(spec.n_draft, room); ++k) {
                            pr *= spec.mtp_acc[k]; e += pr;
                            const double r = (1 + e) / (cost(1 + k) + k * (spec.draft_ms > 0 ? spec.draft_ms : 1.5));
                            if (r > best) { best = r; use_mtp = k; }
                        }
                    }
                    if (n_lk > 0) {
                        const double a = spec.lk_acc[std::min(8, n_lk)];
                        double e = 0, pr = 1;
                        for (int k = 1; k <= (int) lk.size(); ++k) {
                            pr *= a; e += pr;
                            const double r = (1 + e) / cost(1 + k);
                            if (r > best) { best = r; use_lk = k; use_mtp = 0; }
                        }
                    }
                    if (use_lk > 0) {
                        d.assign(lk.begin(), lk.begin() + use_lk);
                    } else if (use_mtp > 0) {
                        std::memcpy(h.data(), pending.data(), E * sizeof(float));
                        llama_token cur = last;
                        for (int k = 0; k < use_mtp; ++k) {
                            llama_batch_ext_clear(spec.db);
                            const int idx = llama_batch_ext_add_token(spec.db, 0, cur);
                            const llama_pos p = pos0 + k;
                            llama_batch_ext_set_pos(spec.db, idx, &p);
                            llama_batch_ext_set_embd_state(spec.db, idx, { h.data(), 1, (size_t) E });
                            llama_batch_ext_set_output_logits(spec.db, idx, true);
                            {
                                std::lock_guard<std::mutex> pk(eng.pass_mu);
                                const double td = now_ms();
                                if (llama_process(spec.dft, LLAMA_PROCESS_TYPE_DECODE, spec.db) != 0) { break; }
                                llama_synchronize(spec.dft);
                                const double ms = now_ms() - td;
                                spec.draft_ms = spec.draft_ms > 0 ? spec.draft_ms * 0.9 + ms * 0.1 : ms;
                            }
                            cur = argmax_row(llama_get_logits_ith(spec.dft, idx), V);
                            d.push_back(cur);
                            std::memcpy(h.data(), llama_get_embeddings_nextn_ith(spec.dft, idx), E * sizeof(float));
                        }
                        llama_memory_seq_rm(md, 0, pos0, -1);
                    }
                    // 3. verify [last, d...] in one pass
                    std::vector<llama_token> vt(1 + d.size());
                    vt[0] = last;
                    for (size_t k = 0; k < d.size(); ++k) { vt[k + 1] = d[k]; }
                    if (!read(vt.data(), (int) vt.size(), pos0, true)) { why = "error"; break; }
                    int j = 0;
                    llama_token y = argmax_row(llama_get_logits_ith(spec.tgt, 0), V);
                    while (j < (int) d.size() && y == d[j]) { ++j; y = argmax_row(llama_get_logits_ith(spec.tgt, j), V); }
                    if (!catch_up(vt.data(), j + 1, pos0)) { why = "error"; break; }
                    if (mtp) { std::memcpy(pending.data(), hrows.data() + (size_t) j * E, E * sizeof(float)); }
                    if (j < (int) d.size() && !llama_memory_seq_rm(mt, 0, pos0 + j + 1, -1)) { why = "error"; break; }
                    // what was kept teaches the choice: the k-th draft counts when the ones before it were kept
                    {
                        double * acc = use_lk > 0 ? &spec.lk_acc[std::min(8, n_lk)] : nullptr;
                        for (int k = 1; k <= (int) d.size() && k <= j + 1; ++k) {
                            double & a = acc ? *acc : spec.mtp_acc[std::min(16, k)];
                            a = a * 0.95 + (k <= j ? 0.05 : 0.0);
                        }
                    }
                    if (use_lk > 0) { spec.n_lk_cycles++; spec.n_lk_kept += j; spec.n_lk_drafted += (long long) d.size(); }
                    if (spec.n_lookup > 0) { hist.insert(hist.end(), vt.begin(), vt.begin() + j + 1); }
                    // 4. out: last and the kept drafts
                    std::string add;
                    bool eos = false;
                    for (int k = 0; k <= j; ++k) {
                        const llama_token t = vt[k];
                        if (llama_vocab_is_eog(eng.vocab, t)) { eos = true; break; }
                        add += eng.piece(t);
                        n_out++;
                    }
                    cycles++; kept += j; drafted += (int) d.size();
                    last = y;
                    pos0 += j + 1;
                    // stop strings: the text ends before the first one
                    size_t cut = std::string::npos;
                    const std::string all = text + add;
                    for (const auto & st : stops) {
                        const size_t f = all.find(st, text.size() > st.size() ? text.size() - st.size() : 0);
                        if (f != std::string::npos && f < cut) { cut = f; }
                    }
                    if (cut != std::string::npos) {
                        if (!emit(all.substr(text.size() < cut ? text.size() : cut, cut > text.size() ? cut - text.size() : 0))) { why = "cancelled"; }
                        else { why = "stop"; }
                        text = all.substr(0, cut);
                        break;
                    }
                    text = all;
                    {
                        std::lock_guard<std::mutex> vk(spec.view_mu);
                        spec.v_tokens = n_out; spec.v_kept = kept; spec.v_drafted = drafted;
                        spec.v_tail = text.size() > 110 ? text.substr(text.size() - 110) : text;
                    }
                    if (!emit(add)) { why = "cancelled"; break; }
                    if (eos) { why = "eos"; break; }
                }
                const double t1 = now_ms();
                { std::lock_guard<std::mutex> vk(spec.view_mu); spec.active = false; }
                {
                    std::lock_guard<std::mutex> lk(eng.snap_mu);
                    eng.recent.push_back(json{{"lane", -1}, {"head", spec.v_head}, {"text", text.substr(0, 120)}, {"tokens", n_out}, {"read", (int) prompt->size()},
                                              {"ms", t1 - t0}, {"stop", why}, {"fork", false}, {"spec", true}, {"kept", kept}, {"drafted", drafted},
                                              {"n", (long long) eng.n_done++}});
                    if (eng.recent.size() > 60) { eng.recent.pop_front(); }
                }
                spec.n_answers++; spec.n_tokens += n_out; spec.n_cycles += cycles; spec.n_kept += kept; spec.n_drafted += drafted;
                json fin = {{"done", json::array({{{"content", text}, {"stop", why}, {"tokens_predicted", n_out}, {"tokens_evaluated", (int) prompt->size()},
                                                   {"session", -1}, {"cycles", cycles}, {"drafts_kept", kept}, {"drafted", drafted}}})},
                            {"timings", {{"read_ms", t_read - t0}, {"total_ms", t1 - t0}, {"tokens_per_second", n_out * 1000.0 / std::max(1.0, t1 - t_read)}}}};
                const std::string line = "data: " + fin.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\ndata: [DONE]\n\n";
                sink.write(line.data(), line.size());
                sink.done();
                return true;
            });
    });

    // what every lane is doing right now, and the last requests that ended (since=<n> gives only newer ones)
    svr.Get("/lanes", [&](const httplib::Request & req, httplib::Response & res) {
        const long long since = req.has_param("since") ? atoll(req.get_param_value("since").c_str()) : -1;
        json lanes = json::array(), rec = json::array();
        {
            std::lock_guard<std::mutex> lk(eng.snap_mu);
            for (size_t i = 0; i < eng.snap.size(); ++i) {
                const auto & p = eng.snap[i];
                if (p.on) { lanes.push_back(json{{"i", (int) i}, {"n", p.n_gen}, {"p", p.n_prompt}, {"f", p.fork}, {"r", p.reader}, {"head", p.head}, {"tail", p.tail}}); }
            }
            for (const auto & r : eng.recent) { if (r.value("n", 0LL) > since) { rec.push_back(r); } }
        }
        json out = {{"lanes", lanes}, {"recent", rec}, {"n_lanes", eng.lanes}, {"width", (int) eng.width}, {"generated_tokens", (long long) eng.n_tokens},
                    {"prompt_tokens", (long long) eng.n_prompt}, {"passes", (long long) eng.n_pass}, {"requests_done", (long long) eng.n_done},
                    {"sessions", (int) eng.n_sessions}, {"forks", (long long) eng.n_forks}};
        if (spec.tgt) {
            std::lock_guard<std::mutex> vk(spec.view_mu);
            out["spec"] = {{"draft", std::max(spec.n_draft, spec.n_lookup)}, {"mtp", spec.n_draft}, {"lookup", spec.n_lookup}, {"active", spec.active}, {"tokens", spec.v_tokens}, {"kept", spec.v_kept}, {"drafted", spec.v_drafted},
                           {"head", spec.v_head}, {"tail", spec.v_tail}, {"total_tokens", (long long) spec.n_tokens}};
            out["generated_tokens"] = (long long) eng.n_tokens + (long long) spec.n_tokens + (spec.active ? spec.v_tokens : 0);
        }
        const double t_now = now_ms();
        out["clock_ms"] = t_now; out["rate"] = rate_now(t_now);                                   // version 4
        // (peak_from=<clock_ms>: the best second since then, what the app's meter shows as an answer's peak while it runs)
        if (req.has_param("peak_from")) { out["peak"] = rate_peak(atof(req.get_param_value("peak_from").c_str()), t_now); }
        res.set_content(out.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    });
    // version 4: GET /speed?from_ms=<a>&to_ms=<b> (the engine's clock, as /stats' clock_ms): the best second between them
    svr.Get("/speed", [&](const httplib::Request & req, httplib::Response & res) {
        const double t_now = now_ms();
        const double a = req.has_param("from_ms") ? atof(req.get_param_value("from_ms").c_str()) : t_now - 60000.0;
        const double b = req.has_param("to_ms") ? atof(req.get_param_value("to_ms").c_str()) : t_now;
        json out = {{"peak", rate_peak(a, b)}, {"rate", rate_now(t_now)}, {"clock_ms", t_now}, {"total_tokens", total_tokens()}};
        res.set_content(out.dump(), "application/json");
    });
    // POST /media {"session": base (optional), "prefix": "...", "image": "<base64>", "suffix": "...", "name": "..."}
    // -> {"session": parked, "tokens": n} : text + image + text read after the base session's state
    svr.Post("/media", [&](const httplib::Request & req, httplib::Response & res) {
        json body = json::parse(req.body, nullptr, false);
        if (!eng.mctx || body.is_discarded() || !body.contains("image") || !body["image"].is_string()) {
            res.status = 400; res.set_content("{\"error\":\"no vision projector loaded (--mmproj), or no image\"}", "application/json"); return;
        }
        const std::string b64 = body["image"].get<std::string>();
        std::vector<unsigned char> bytes;
        {   // base64
            static const std::string tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            int val = 0, bits = -8;
            for (unsigned char c : b64) {
                const size_t k = tbl.find(c);
                if (k == std::string::npos) { continue; }
                val = (val << 6) + (int) k; bits += 6;
                if (bits >= 0) { bytes.push_back((unsigned char) ((val >> bits) & 0xFF)); bits -= 8; }
            }
        }
        const int base = body.value("session", -1);
        const std::string text = body.value("prefix", std::string()) + mtmd_default_marker() + body.value("suffix", std::string());
        const std::string name = body.value("name", std::string());
        auto prom = std::make_shared<std::promise<json>>();
        auto fut = prom->get_future();
        {
            std::lock_guard<std::mutex> lk(eng.job_mu);
            eng.jobs.push_back([&eng, prom, bytes, base, text, name]() {
                json out;
                int sl = -1;
                llama_pos p0 = 0;
                {
                    std::lock_guard<std::mutex> lk(eng.mu);
                    if (base >= 0 && (base >= (int) eng.vault.size() || !eng.vault[base].used || eng.vault[base].pending >= 0)) {
                        prom->set_value(json{{"error", "the base session is not a parked state that was read"}}); return;
                    }
                    for (size_t k = 0; k < eng.vault.size(); ++k) { if (!eng.vault[k].used) { sl = (int) k; break; } }
                    if (sl >= 0) { eng.vault[sl] = Engine::Slot(); eng.vault[sl].used = true; eng.vault[sl].user = true; eng.vault[sl].name = name; }
                    if (base >= 0) { p0 = eng.vault[base].pos; }
                }
                if (sl < 0) { prom->set_value(json{{"error", "no free session slot (start rapidli-engine with --vault <n>)"}}); return; }
                const int32_t seq = eng.lanes + sl;
                llama_memory_t mem = llama_get_memory(eng.ctx);
                bool ok = true;
                if (base >= 0) { ok = llama_fq_fork(eng.ctx, eng.lanes + base, &seq, 1) != 0; }
                else { llama_memory_seq_rm(mem, seq, -1, -1); }
                mtmd_helper_bitmap_wrapper bw = ok ? mtmd_helper_bitmap_init_from_buf(eng.mctx, bytes.data(), bytes.size(), false, mtmd_helper_init_opt_default())
                                                   : mtmd_helper_bitmap_wrapper{nullptr, nullptr};
                int n_tok = 0;
                llama_pos np = p0;
                std::string err;
                if (!ok) { err = "the fork failed"; }
                else if (!bw.bitmap) { err = "the image could not be read (jpg, png, bmp or gif)"; }
                else {
                    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
                    mtmd_input_text it{text.c_str(), text.size(), false, true};
                    const mtmd_bitmap * bm = bw.bitmap;
                    if (mtmd_tokenize(eng.mctx, chunks, &it, &bm, 1) != 0) { err = "the image could not be tokenized"; }
                    else {
                        n_tok = (int) mtmd_helper_get_n_tokens(chunks);
                        std::lock_guard<std::mutex> pk(eng.pass_mu);
                        if (mtmd_helper_eval_chunks(eng.mctx, eng.ctx, chunks, p0, seq, eng.chunk, false, &np) != 0) { err = "reading the image failed"; }
                    }
                    mtmd_input_chunks_free(chunks);
                    mtmd_bitmap_free(bw.bitmap);
                }
                if (!err.empty()) {
                    llama_memory_seq_rm(mem, seq, -1, -1);
                    std::lock_guard<std::mutex> lk(eng.mu);
                    eng.vault[sl] = Engine::Slot();
                    prom->set_value(json{{"error", err}});
                    return;
                }
                {
                    std::lock_guard<std::mutex> lk(eng.mu);
                    eng.vault[sl].pos = np; eng.vault[sl].pending = -1;
                }
                eng.n_sessions++;
                eng.n_prompt += n_tok;
                prom->set_value(json{{"session", sl}, {"tokens", n_tok}, {"pos", (int) np}, {"stop", "read"}});
            });
        }
        eng.wake = true;
        eng.cv.notify_one();
        json out = fut.get();
        if (out.contains("error")) { res.status = 400; }
        res.set_content(out.dump(), "application/json");
    });
    // POST /tokenize {"text": "..."} -> {"n": tokens}
    svr.Post("/tokenize", [&](const httplib::Request & req, httplib::Response & res) {
        json body = json::parse(req.body, nullptr, false);
        const std::string text = (!body.is_discarded() && body.contains("text") && body["text"].is_string()) ? body["text"].get<std::string>() : std::string();
        res.set_content(json({{"n", (int) eng.tokenize(text).size()}}).dump(), "application/json");
    });
    // The card warmed for {"ms": n} (at most 2,000) while the user types, see Warmer
    svr.Post("/warm", [&](const httplib::Request & req, httplib::Response & res) {
        json body = json::parse(req.body, nullptr, false);
        const int ms = (!body.is_discarded() && body.contains("ms") && body["ms"].is_number_integer()) ? body["ms"].get<int>() : 400;
        g_warmer.warm(ms);
        res.set_content("{\"ok\":true}", "application/json");
    });
    svr.Post("/release", [&](const httplib::Request & req, httplib::Response & res) {
        json body = json::parse(req.body, nullptr, false);
        const bool ok = !body.is_discarded() && body.contains("session") && body["session"].is_number_integer() && eng.release(body["session"].get<int>());
        res.set_content(ok ? "{\"ok\":true}" : "{\"ok\":false}", "application/json");
    });
    // One text read once, then many continuations of it written at the same time:
    //   {"session": s (optional: start from a parked state), "prompt": "text every branch shares" (optional with a session),
    //    "branches": ["text only branch 0 reads", ...], "n_predict": n, "stop": [...], "hold": false, "release": false,
    //    "temperature", "min_p", "seed"}
    // Answers with server-sent events: {"rows": [[branch, "text"], ...]} as the branches write, then
    // {"done": [the result of every branch], "timings": {...}} and [DONE].
    svr.Post("/fork", [&](const httplib::Request & req, httplib::Response & res) {
        json body = json::parse(req.body, nullptr, false);
        auto fail = [&](const std::string & e) { res.status = 400; res.set_content(json({{"error", e}}).dump(), "application/json"); };
        if (body.is_discarded() || !body.contains("branches") || !body["branches"].is_array() || body["branches"].empty()) { fail("branches missing"); return; }
        const double t0 = now_ms();
        const int session = body.contains("session") && body["session"].is_number_integer() ? body["session"].get<int>() : -1;
        const std::string common = body.value("prompt", std::string());
        if (session < 0 && common.empty()) { fail("prompt or session missing"); return; }
        std::string err;
        int src = session;
        double read_ms = 0;
        if (!common.empty()) {
            json b0 = {{"hold", true}, {"bulk", true}, {"read_only", true}};
            if (session >= 0) { b0["session"] = session; }
            json out0;
            if (!submit(eng.tokenize(common), 1, out0, err, b0)) { fail(err); return; }
            src = out0.value("session", -1);
            if (src < 0) { fail("no free session slot (start rapidli-engine with --vault <n>)"); return; }
            read_ms = now_ms() - t0;
        }
        const size_t n = body["branches"].size();
        auto g    = std::make_shared<Group>(n);
        auto reqs = std::make_shared<std::vector<std::shared_ptr<Request>>>();
        json bk = body;
        bk.erase("prompt"); bk.erase("branches"); bk.erase("read_only");
        bk["session"] = src;
        bk["bulk"]    = body.value("bulk", false);
        const bool seeded = body.contains("seed") && body["seed"].is_number_integer();
        for (size_t k = 0; k < n; ++k) {
            if (seeded) { bk["seed"] = body["seed"].get<long long>() + (long long) k; }
            std::string e;
            const json & bj = body["branches"][k];          // a text, or {"text": ..., and options of this branch only}
            json bo = bk;
            std::string text;
            if (bj.is_string()) { text = bj.get<std::string>(); }
            else if (bj.is_object()) {
                text = bj.value("text", std::string());
                for (auto it = bj.begin(); it != bj.end(); ++it) { if (it.key() != "text" && it.key() != "session") { bo[it.key()] = it.value(); } }
            }
            auto r = enqueue(eng.tokenize(text), bo.value("n_predict", bo.value("max_tokens", 200)), true, e, bo, g, (int) k);
            if (r) { reqs->push_back(r); std::lock_guard<std::mutex> lk(g->mu); g->members.push_back(r); }
            else {
                std::lock_guard<std::mutex> lk(g->mu);
                g->finals[k] = json({{"error", e}, {"content", ""}, {"stop", "error"}, {"tokens_predicted", 0}, {"tokens_evaluated", 0}});
                g->n_done++;
            }
        }
        if (!common.empty()) { eng.release(src); }                           // freed once every branch has started
        if (body.value("release", false) && session >= 0) { eng.release(session); }
        auto buf = std::make_shared<std::vector<std::string>>(n);
        res.set_chunked_content_provider("text/event-stream",
            [g, n, t0, read_ms, buf](size_t, httplib::DataSink & sink) {
                bool done; json fin;
                {
                    std::unique_lock<std::mutex> lk(g->mu);
                    g->cv.wait(lk, [&] {
                        if (g->n_done >= (int) n) { return true; }
                        for (const auto & f : g->fresh) { if (!f.empty()) { return true; } }
                        return false;
                    });
                    for (size_t k = 0; k < n; ++k) { (*buf)[k] += g->fresh[k]; g->fresh[k].clear(); }
                    done = g->n_done >= (int) n;
                    if (done) { fin = g->finals; }
                }
                json rows = json::array();
                for (size_t k = 0; k < n; ++k) {
                    std::string & b = (*buf)[k];
                    const size_t w = done ? b.size() : utf8_whole(b);
                    if (w > 0) { rows.push_back(json::array({(int) k, b.substr(0, w)})); b.erase(0, w); }
                }
                if (!rows.empty()) {
                    const std::string line = "data: " + json({{"rows", rows}}).dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                    if (!sink.write(line.data(), line.size())) { return false; }
                }
                if (done) {
                    json c = {{"done", fin}, {"timings", {{"read_ms", read_ms}, {"total_ms", now_ms() - t0}}}};
                    const std::string line = "data: " + c.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\ndata: [DONE]\n\n";
                    sink.write(line.data(), line.size());
                    sink.done();
                }
                return true;
            },
            [reqs](bool success) { if (!success) { for (auto & r : *reqs) { r->cancelled = true; } } });
    });
    svr.Post("/completion", [&](const httplib::Request & req, httplib::Response & res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("prompt")) { res.status = 400; res.set_content("{\"error\":\"prompt missing\"}", "application/json"); return; }
        json out; std::string err;
        if (body.value("hold", false) && body.contains("name") && body["name"].is_string()) {     // version 2: parked under this name already?
            const std::string name = body["name"].get<std::string>();
            std::lock_guard<std::mutex> lk(eng.mu);
            for (size_t k = 0; k < eng.vault.size(); ++k) {
                if (eng.vault[k].used && !eng.vault[k].freeing && eng.vault[k].user && !name.empty() && eng.vault[k].name == name) {
                    res.set_content(json({{"session", (int) k}, {"cached", true}, {"content", ""}, {"stop", "read"}}).dump(), "application/json");
                    return;
                }
            }
            if (body.value("lookup_only", false)) {    // version 2: only asking whether it is parked (a chat's conversation)
                res.set_content(json({{"session", -1}, {"cached", false}}).dump(), "application/json");
                return;
            }
        }
        if (!submit(eng.tokenize(body["prompt"].get<std::string>()), body.value("n_predict", body.value("max_tokens", 200)), out, err, body)) {
            res.status = 400; res.set_content(json({{"error", err}}).dump(), "application/json"); return;
        }
        res.set_content(out.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    });
    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("messages")) { res.status = 400; res.set_content("{\"error\":\"messages missing\"}", "application/json"); return; }
        bool thinking = true;
        if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].contains("enable_thinking")) {
            thinking = body["chat_template_kwargs"]["enable_thinking"].get<bool>();
        }
        json out; std::string err;
        if (body.value("stream", false)) {
            // server-sent events in the OpenAI format: one "data: {chunk}" per piece of text, then "data: [DONE]"
            auto r = enqueue(eng.tokenize(chatml(body["messages"], thinking)), body.value("max_tokens", 200), true, err, body);
            if (!r) { res.status = 400; res.set_content(json({{"error", err}}).dump(), "application/json"); return; }
            struct State { std::string buf; bool in_think; bool trim; };
            auto st = std::make_shared<State>(State{std::string(), thinking, false});
            res.set_chunked_content_provider("text/event-stream",
                [r, st](size_t, httplib::DataSink & sink) {
                    std::string add; bool done; json fin;
                    {
                        std::unique_lock<std::mutex> lk(r->smu);
                        r->scv.wait(lk, [&] { return !r->fresh.empty() || r->sdone; });
                        add.swap(r->fresh);
                        done = r->sdone;
                        if (done) { fin = r->final; }
                    }
                    st->buf += add;
                    auto emit = [&](const char * field, const std::string & text) {
                        if (text.empty()) { return true; }
                        json c = {{"object", "chat.completion.chunk"}, {"model", g_model_id},
                                  {"choices", json::array({{{"index", 0}, {"delta", {{field, text}}}, {"finish_reason", nullptr}}})}};
                        const std::string line = "data: " + c.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
                        return sink.write(line.data(), line.size());
                    };
                    bool ok = true;
                    if (st->in_think) {
                        const size_t e = st->buf.find("</think>");
                        if (e != std::string::npos) {
                            ok = emit("reasoning_content", st->buf.substr(0, e));
                            st->buf.erase(0, e + 8);
                            st->in_think = false;
                            st->trim     = true;
                        } else {
                            // keep back what could be the beginning of the closing tag, and half a character
                            size_t keep = 0;
                            for (size_t k = std::min<size_t>(7, st->buf.size()); k > 0; --k) {
                                if (st->buf.compare(st->buf.size() - k, k, "</think>", k) == 0) { keep = k; break; }
                            }
                            std::string head = st->buf.substr(0, st->buf.size() - keep);
                            const size_t w = done ? head.size() : utf8_whole(head);
                            ok = emit("reasoning_content", head.substr(0, w));
                            st->buf.erase(0, w);
                        }
                    }
                    if (ok && !st->in_think) {
                        if (st->trim) {
                            size_t k = 0;
                            while (k < st->buf.size() && st->buf[k] == '\n') { ++k; }
                            st->buf.erase(0, k);
                            if (!st->buf.empty()) { st->trim = false; }
                        }
                        const size_t w = done ? st->buf.size() : utf8_whole(st->buf);
                        ok = emit("content", st->buf.substr(0, w));
                        st->buf.erase(0, w);
                    }
                    if (!ok) { r->cancelled = true; return false; }
                    if (done) {
                        json c = {{"object", "chat.completion.chunk"}, {"model", g_model_id},
                                  {"choices", json::array({{{"index", 0}, {"delta", json::object()}, {"finish_reason", fin["stop"] == "limit" ? "length" : "stop"}}})},
                                  {"usage", {{"prompt_tokens", fin["tokens_evaluated"]}, {"completion_tokens", fin["tokens_predicted"]},
                                             {"total_tokens", fin["tokens_evaluated"].get<int>() + fin["tokens_predicted"].get<int>()}}},
                                  {"timings", fin["timings"]}};
                        const std::string line = "data: " + c.dump() + "\n\ndata: [DONE]\n\n";
                        sink.write(line.data(), line.size());
                        sink.done();
                    }
                    return true;
                },
                [r](bool success) { if (!success) { r->cancelled = true; } });
            return;
        }
        if (!submit(eng.tokenize(chatml(body["messages"], thinking)), body.value("max_tokens", 200), out, err, body)) {
            res.status = 400; res.set_content(json({{"error", err}}).dump(), "application/json"); return;
        }
        std::string content = out["content"].get<std::string>(), reasoning;
        if (thinking) {
            const size_t e = content.find("</think>");
            if (e != std::string::npos) {
                reasoning = content.substr(0, e);
                content   = content.substr(e + 8);
                while (!content.empty() && content[0] == '\n') { content.erase(0, 1); }
            }
        }
        json r = {
            {"object", "chat.completion"},
            {"model", g_model_id},
            {"choices", json::array({{{"index", 0}, {"finish_reason", out["stop"] == "limit" ? "length" : "stop"},
                                      {"message", {{"role", "assistant"}, {"content", content}, {"reasoning_content", reasoning}}}}})},
            {"usage", {{"prompt_tokens", out["tokens_evaluated"]}, {"completion_tokens", out["tokens_predicted"]},
                       {"total_tokens", out["tokens_evaluated"].get<int>() + out["tokens_predicted"].get<int>()}}},
            {"timings", out["timings"]},
        };
        res.set_content(r.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    });

    fprintf(stderr, "engine: %d lanes, %d tokens of context per lane, %d-bit state, %s, listening on 127.0.0.1:%d\n", lanes, ctx_lane,
            getenv("LLAMA_RS_I8") ? 8 : 16, use_chains ? (temp > 0.0f ? "sampler chains" : "greedy chains") : "sampling per request (default greedy)", port);
    fflush(stderr);
    svr.listen("127.0.0.1", port);

    eng.stop = true;
    eng.cv.notify_all();
    worker.join();
    speed_stop = true;
    speed_th.join();
    return 0;
}
