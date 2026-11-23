"""Patch 1 for llama.dll (Rapidli): read each stream's recurrent state straight from the state cache.

Stock behaviour: on every pass, every gated-delta-net block gathers the states of the streams in the batch out of the
cache (ggml_get_rows: one full read and write of 3.1 MB per stream per block), the kernel updates them, and the result
is written back. When every stream's state already sits in its own cell (cell head+i holds stream i) the gather is a
copy of the cache onto itself. The patch hands the kernel a VIEW of the cache instead; the CUDA kernel reads its part of
the state into registers before it writes that same part back, so the update is in place.
A stream that starts (its state must be zero) is handled too: its cell is cleared before the graph runs, instead of
being copied from a cleared cell inside the graph.
One token per stream per pass; with a 16-bit state cache (patch_f16.py) also runs of up to 64 tokens per stream, because
the kernel used then keeps the state in registers for the whole run (the float32 path switches to another kernel for
runs of tokens, which has not been checked for in-place use).
Opt-in: environment variable LLAMA_RS_DIRECT=1.
  python patch_direct.py        (run once on the pristine source in src/)"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
NL = chr(10)


def patch(name, pairs):
    p = os.path.join(HERE, "src", name)
    s = open(p, encoding="utf8").read().replace(chr(13) + NL, NL)
    for pair in pairs:
        a, b, cnt = pair[0], pair[1], (pair[2] if len(pair) > 2 else 1)
        assert s.count(a) == cnt, (name, a[:70], s.count(a))
        s = s.replace(a, b)
    open(p, "w", encoding="utf8", newline=NL).write(s)


patch("llama-graph.h", [(
    """    // used in view offsets, need to match for valid graph reuse
    uint32_t head;
    int32_t rs_z;
};""",
    """    // used in view offsets, need to match for valid graph reuse
    uint32_t head;
    int32_t rs_z;

    // RAPIDLI: the states of the ubatch are read straight from the cache (a view, no gather); see llm_rs_direct()
    bool direct = false;
    // RAPIDLI: the graph clears and reads the "zero" cell rs_z (the gather path does); a graph of direct views does
    // not, so it stays valid when another stream starts
    bool rs_z_used = false;
};

// RAPIDLI: the recurrent state cache is kept in 16-bit floats (LLAMA_RS_F16=1, patch_f16.py)
bool llm_rs_f16();
// RAPIDLI: true when every state of the ubatch sits in its own cache cell (or starts from zero) and can be updated in place
bool llm_rs_direct(const llama_ubatch & ubatch, const llama_memory_recurrent_context * mctx, uint32_t n_rs_seq);
// RAPIDLI: clears the cells of starting sequences before the graph runs (what the graph's own clear cannot do)
void llm_rs_prepare(const llama_memory_recurrent_context * mctx, bool direct);""")])

patch("llama-memory-recurrent.h", [(
    """    ggml_tensor * get_s_l(int32_t il) const;""",
    """    ggml_tensor * get_s_l(int32_t il) const;
    int32_t get_n_s_l() const; // RAPIDLI: number of layers in the state cache""")])

patch("llama-memory-recurrent.cpp", [(
    """ggml_tensor * llama_memory_recurrent_context::get_s_l(int32_t il) const {""",
    """int32_t llama_memory_recurrent_context::get_n_s_l() const {
    return (int32_t) mem->s_l.size();
}

ggml_tensor * llama_memory_recurrent_context::get_s_l(int32_t il) const {""")])

patch("llama-graph.cpp", [
    ("""    res &= s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}
""",
     """    res &= s_copy_main->ne[0] == params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= !rs_z_used || rs_z == mctx->get_rs_z();

    res &= direct == llm_rs_direct(params.ubatch, mctx, params.cparams.n_rs_seq);

    return res;
}

bool llm_rs_f16() {
    static const bool enabled = getenv("LLAMA_RS_F16") != nullptr && getenv("LLAMA_RS_F16")[0] == '1';
    return enabled;
}

bool llm_rs_direct(const llama_ubatch & ubatch, const llama_memory_recurrent_context * mctx, uint32_t n_rs_seq) {
    static const bool enabled = getenv("LLAMA_RS_DIRECT") != nullptr && getenv("LLAMA_RS_DIRECT")[0] == '1';
    if (!enabled || n_rs_seq != 0) {
        return false;
    }
    // one token per sequence; with a 16-bit cache also short runs (that kernel keeps the state in registers throughout)
    if (ubatch.n_seq_tokens != 1 && !(llm_rs_f16() && ubatch.n_seq_tokens <= 64)) {
        return false;
    }
    const uint32_t n_rs = mctx->get_n_rs();
    const uint32_t head = mctx->get_head();
    const int32_t  rs_z = mctx->get_rs_z();
    if (n_rs != ubatch.n_seqs) {
        return false;
    }
    for (uint32_t i = 0; i < n_rs; ++i) {
        const int32_t src = mctx->s_copy(i);   // n_rs_seq == 0: s_copy has no side effect
        if (src != (int32_t) (head + i) && !(rs_z >= 0 && src == rs_z)) {   // its own cell, or a start from zero
            return false;
        }
    }
    return true;
}

void llm_rs_prepare(const llama_memory_recurrent_context * mctx, bool direct) {
    const int32_t rs_z = mctx->get_rs_z();
    if (rs_z < 0) {
        return;
    }
    const uint32_t head = mctx->get_head();
    const uint32_t n_rs = mctx->get_n_rs();
    // direct: the cells of the starting sequences, as runs of neighbouring cells [first, first + count)
    std::vector<std::pair<uint32_t, uint32_t>> runs;
    if (direct) {
        for (uint32_t i = 0; i < n_rs; ++i) {
            if (mctx->s_copy(i) != rs_z) {
                continue;
            }
            if (!runs.empty() && runs.back().first + runs.back().second == head + i) {
                runs.back().second++;
            } else {
                runs.push_back({ head + i, 1 });
            }
        }
    }
    for (int32_t il = 0; il < mctx->get_n_s_l(); ++il) {
        for (ggml_tensor * t : { mctx->get_r_l(il), mctx->get_s_l(il) }) {
            if (t == nullptr || t->buffer == nullptr) {
                continue;
            }
            if (direct) {
                for (const auto & run : runs) {
                    ggml_backend_tensor_memset(t, 0, (size_t) run.first*t->nb[1], (size_t) run.second*t->nb[1]);
                }
            } else if (t->type != GGML_TYPE_F32) {
                // the gather path: the graph clears the zero cell of a float32 cache itself, not of a 16-bit one
                ggml_backend_tensor_memset(t, 0, (size_t) rs_z*t->nb[1], t->nb[1]);
            }
        }
    }
}
"""),
    ("""    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);
""",
     """    const llama_memory_recurrent_context * mctx_cur,
               uint32_t   n_rs_seq) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);
    inp->direct = llm_rs_direct(ubatch, mctx_cur, n_rs_seq);
"""),
    ("""    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));""",
     """    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur, cparams.n_rs_seq);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));"""),
    ("""    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());""",
     """    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr(), cparams.n_rs_seq);""", 2),
    ("""    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());""",
     """    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr(), cparams.n_rs_seq);"""),
    ("""    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();
""",
     """    res &= !inp_rs->rs_z_used || inp_rs->rs_z == mctx->get_recr()->get_rs_z();
    res &= inp_rs->direct == llm_rs_direct(params.ubatch, mctx->get_recr(), params.cparams.n_rs_seq);
""", 3),
    ("""#include <algorithm>
#include <cassert>
#include <cmath>
""",
     """#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>
"""),
    # with a direct view the copy indices are not part of the graph, so they have no buffer to fill
    ("""    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));""",
     """    llm_rs_prepare(mctx, direct);
    if (s_copy && s_copy->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));"""),
    ("""    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));""",
     """    llm_rs_prepare(mctx->get_recr(), inp_rs->direct);
    if (inp_rs->s_copy && inp_rs->s_copy->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));""", 3),
    ("""    const auto * kv_state = inp->mctx;

    // a custom getter reads the states of the ubatch straight from the cache, so the gather skips the first""",
     """    const auto * kv_state = inp->mctx;

    // RAPIDLI: every state of the ubatch is already in its own cell (cell head+i = sequence i): a view of the cache
    if (inp->direct && !get_state_rows) {
        ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);
        return ggml_view_2d(ctx0, states, state_size, n_seqs, states->nb[1], (size_t) kv_state->get_head()*states->nb[1]);
    }
    inp->rs_z_used = true;

    // a custom getter reads the states of the ubatch straight from the cache, so the gather skips the first"""),
    # the graph's own clear of the zero cell works on float32 caches only (a 16-bit one is cleared in llm_rs_prepare)
    ("""    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));
""",
     """    if (states->type == GGML_TYPE_F32) {
        ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));
    }
"""),
])
# The order of the streams inside a pass follows the order of the tokens in the batch, and the server lists generating
# slots first and slots that read a prompt last. The recurrent memory keeps the cells in pass order, so every such
# change of order moved whole states around (and left the direct path for two passes). With LLAMA_RS_DIRECT=1 the
# streams of a pass are sorted by id, so a stream stays in its cell.
patch("llama-batch.cpp", [(
    """#include <cassert>
#include <cstring>
""",
    """#include <cassert>
#include <cstdlib>
#include <cstring>
"""), (
    """    uint32_t n_seqs = cur_seq_set.size();

    // we are done
    if (n_seqs == 0) {
        return {};
    }

    // the current batch index of each sequence set
    std::vector<int32_t> cur_idx(n_seqs, 0);
""",
    """    uint32_t n_seqs = cur_seq_set.size();

    // we are done
    if (n_seqs == 0) {
        return {};
    }

    // RAPIDLI: streams in id order (only when no token belongs to several sequences)
    {
        static const bool sorted = getenv("LLAMA_RS_DIRECT") != nullptr && getenv("LLAMA_RS_DIRECT")[0] == '1';
        if (sorted && !sequential && !has_cpl) {
            auto first_id = [](const seq_set_t & set) {
                for (size_t b = 0; b < set.size(); ++b) {
                    if (set.test(b)) {
                        return b;
                    }
                }
                return set.size();
            };
            std::stable_sort(cur_seq_set.begin(), cur_seq_set.end(), [&](const seq_set_t & a, const seq_set_t & b) {
                return first_id(a) < first_id(b);
            });
        }
    }

    // the current batch index of each sequence set
    std::vector<int32_t> cur_idx(n_seqs, 0);
""")])
print("patched: direct recurrent state (llama-graph, llama-memory-recurrent), streams in id order (llama-batch)")
