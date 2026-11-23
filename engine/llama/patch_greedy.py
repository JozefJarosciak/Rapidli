"""Patch for llama.dll (Rapidli): one sampling operation for all streams instead of a sampler chain per stream.

With sampling on the card every stream gets its own chain of small operations in the graph (5 for the shortest greedy
chain, about 28 for the server's default chain) and three small copies to the host per pass. With LLAMA_FAST_GREEDY=1
(set by rapidli-engine) the graph takes ONE operation over all rows of logits and one copy brings all sampled tokens to
the host.

That operation is the arg max, with a small input "fq_sample_params" [4, rows] attached as a second operand. Our
ggml-cuda.dll (cuda-build/patch_sample.py) then draws per row: arg max when the row's temperature is 0, otherwise one
draw from softmax(logits / temperature) restricted to the tokens with probability >= min_p * the largest probability
(probabilities before the temperature, as in llama.cpp's default sampler order)
(Gumbel-max trick, random numbers from a hash on the card). The settings are per sequence and can change at any time
without rebuilding the graph:
    void llama_fq_set_sampling(int32_t seq, float temperature, float min_p, uint32_t seed);     (exported, C linkage)
A backend that ignores the second operand (the CPU) simply returns the arg max.
  python patch_greedy.py        (run after patch_direct.py / patch_f16.py on the source in src/)"""
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
    """    std::vector<ggml_tensor *> t_sampled;
""",
    """    std::vector<ggml_tensor *> t_sampled;
    ggml_tensor * t_sampled_rows = nullptr; // RAPIDLI: the sampled token of every output row, from one operation
""")])

patch("llama-graph.cpp", [
    ("""    t_sampled.clear();
""",
     """    t_sampled.clear();
    t_sampled_rows = nullptr;
"""),
    ("""void llm_graph_context::build_sampling() const {""",
     """// RAPIDLI: sampling settings per sequence for the fused sampler (temperature 0 = arg max), set by the application
struct fq_sample_cfg {
    float    temp  = 0.0f;
    float    min_p = 0.0f;
    uint32_t seed  = 0;
    uint32_t draws = 0;
};
static fq_sample_cfg fq_sample_table[LLAMA_MAX_SEQ];

#ifndef FQ_EXPORT
#if defined(_WIN32)
#  define FQ_EXPORT extern "C" __declspec(dllexport)
#else
#  define FQ_EXPORT extern "C" __attribute__((visibility("default")))
#endif
#endif
FQ_EXPORT void llama_fq_set_sampling(int32_t seq, float temp, float min_p, uint32_t seed) {
    if (seq < 0 || seq >= LLAMA_MAX_SEQ) {
        return;
    }
    fq_sample_table[seq].temp  = temp;
    fq_sample_table[seq].min_p = min_p;
    fq_sample_table[seq].seed  = seed;
    fq_sample_table[seq].draws = 0;
}

class llm_graph_input_fq_sample : public llm_graph_input_i {
public:
    void set_input(const llama_ubatch * ubatch) override {
        const int64_t n_rows = params->ne[1];
        std::vector<float> data(4 * n_rows, 0.0f);
        int64_t r = 0;
        for (uint32_t i = 0; i < ubatch->n_tokens && r < n_rows; ++i) {
            if (!ubatch->output[i]) {
                continue;
            }
            const llama_seq_id s = ubatch->seq_id[i][0];
            if (s >= 0 && s < LLAMA_MAX_SEQ && fq_sample_table[s].temp > 0.0f) {
                fq_sample_cfg & c = fq_sample_table[s];
                data[4*r + 0] = 1.0f / c.temp;
                data[4*r + 1] = c.min_p > 0.0f ? -logf(c.min_p) : 1.0e30f;   // on the probabilities before the temperature, as llama.cpp's min-p
                uint32_t z = c.seed * 0x9E3779B1U + (c.draws++) * 0x85EBCA6BU + 0x165667B1U;
                z ^= z >> 16; z *= 0x7feb352dU; z ^= z >> 15; z *= 0x846ca68bU; z ^= z >> 16;
                memcpy(&data[4*r + 2], &z, sizeof(z));
            }
            r++;
        }
        ggml_backend_tensor_set(params, data.data(), 0, ggml_nbytes(params));
    }

    bool can_reuse(const llm_graph_params & p) override {
        return params != nullptr && params->ne[1] == (int64_t) p.n_outputs;
    }

    ggml_tensor * params = nullptr; // F32 [4, n_outputs]
};

void llm_graph_context::build_sampling() const {"""),
    ("""    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);""",
     """    // RAPIDLI: one operation samples every row (LLAMA_FAST_GREEDY=1)
    {
        static const bool fast_greedy = getenv("LLAMA_FAST_GREEDY") != nullptr && getenv("LLAMA_FAST_GREEDY")[0] == '1';
        if (fast_greedy) {
            ggml_tensor * rows = ggml_argmax(ctx0, res->t_logits);   // I32 [n_outputs]
            ggml_set_name(rows, "sampled_rows");

            auto inp_fq = std::make_unique<llm_graph_input_fq_sample>();
            inp_fq->params = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 4, res->t_logits->ne[1]);
            ggml_set_input(inp_fq->params);
            ggml_set_name(inp_fq->params, "fq_sample_params");
            rows->src[1] = inp_fq->params;
            res->add_input(std::move(inp_fq));

            ggml_set_output(rows);
            ggml_build_forward_expand(gf, rows);
            res->t_sampled_rows = rows;
            res->t_sampling_all.push_back(rows);
            return;
        }
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);"""),
])

patch("llama-context.cpp", [(
    """            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
""",
    """            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            if (res->t_sampled_rows != nullptr && sampling.sampled.has_data()) {
                // RAPIDLI: all rows in one copy
                ggml_backend_t backend_s = ggml_backend_sched_get_tensor_backend(sched.get(), res->t_sampled_rows);
                GGML_ASSERT((size_t) n_outputs_prev + (size_t) ggml_nelements(res->t_sampled_rows) <= (size_t) sampling.sampled.size);
                ggml_backend_tensor_get_async(backend_s, res->t_sampled_rows, sampling.sampled.data + n_outputs_prev, 0, ggml_nbytes(res->t_sampled_rows));
            }
""")])
print("patched: one sampling operation for all streams (llama-graph, llama-context)")
