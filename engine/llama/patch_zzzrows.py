"""Patch for our copy of llama.cpp (src/): Inkling 2's attention over each sequence's own cells.

When many lanes write at once (a ubatch of several sequences), Inkling's relative-position bias took the dense path: a
bias of n_kv x n_tokens x n_head per layer over the whole unified cache (every lane's cells), built, copied and added to
the mask on every pass. On Inkling-2-Mini Q2_K_XL with 62 lanes that was ~290 ms of a ~400 ms pass (23 Nov, measured
with GGML_CUDA_OPTIME), and the pass time grew with lanes x cells.

Now, for a ubatch that splits into per-sequence slices (the lanes' one-token passes), the masks are built over the
kv_rows cell lists of each sequence (as every other model does with fused_kv_rows), the distances are filled over those
lists, and the bias (n_kv_rows x tokens x n_head x slices) is folded into the per-head mask of the kv_rows flash
attention. Prompt reading (one sequence) keeps the banded path; other models are not touched. Needs the ggml-cuda
patch engine/cuda/patch_zrowsmask.py (a per-head mask with kv_rows). LLAMA_REL_ROWS_DISABLE=1 turns it off.
  python patch_zzzrows.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
NL = chr(10)


def load(name):
    return open(os.path.join(SRC, name), encoding="utf8").read().replace(chr(13) + NL, NL)


def save(name, s):
    open(os.path.join(SRC, name), "w", encoding="utf8", newline=NL).write(s)


def rep(s, a, b, name):
    assert s.count(a) == 1, (name, a[:80], s.count(a))
    return s.replace(a, b)


# --- llama-kv-cache.h / .cpp: the distances over the kv_rows cell lists ---------------------------------------------
n = "llama-kv-cache.h"
s = load(n)
if "set_input_relative_pos_rows" in s:
    raise SystemExit("already patched")
decl = "    void set_input_relative_pos(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const;" + NL
assert s.count(decl) == 2, s.count(decl)
s = s.replace(decl, decl + "    // RAPIDLI: the same over the kv_rows cell lists of each slice: dst [n_kv_rows, n_tokens]" + NL
              + "    void set_input_relative_pos_rows(ggml_tensor * dst, const ggml_tensor * rows, const llama_ubatch * ubatch, uint32_t extent) const;" + NL)
save(n, s)

n = "llama-kv-cache.cpp"
s = load(n)
s = rep(s, "void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {",
        """// RAPIDLI: distances from each token to the cells of its slice's kv_rows list (see set_input_relative_pos); padding
// entries (-1) and cells farther than extent get extent, like empty cells
void llama_kv_cache::set_input_relative_pos_rows(ggml_tensor * dst, const ggml_tensor * rows, const llama_ubatch * ubatch, uint32_t extent) const {
    GGML_ASSERT(dst->type == GGML_TYPE_I32 && n_stream == 1);
    GGML_ASSERT(rows && rows->buffer && ggml_backend_buffer_is_host(rows->buffer));
    const int64_t n_rows   = dst->ne[0];
    const int64_t n_slices = rows->ne[1];
    GGML_ASSERT(rows->ne[0] == n_rows && dst->ne[1] == ubatch->n_tokens && ubatch->n_tokens % n_slices == 0);
    const int64_t n_tps = ubatch->n_tokens/n_slices;
    const int32_t * r = (const int32_t *) rows->data;
    const auto & cells = v_cells[0];
    std::vector<int32_t> data(n_rows * ubatch->n_tokens);
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const int32_t * rs = r + (i/n_tps)*n_rows;
        for (int64_t j = 0; j < n_rows; ++j) {
            const int32_t c = rs[j];
            const int64_t distance = c < 0 || cells.is_empty(c) ? -1 : int64_t(ubatch->pos[i]) - cells.pos_get(c);
            data[i*n_rows + j] = distance >= 0 && distance < extent ? distance : extent;
        }
    }
    ggml_backend_tensor_set(dst, data.data(), 0, ggml_nbytes(dst));
}

void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {""", n)
s = rep(s, """void llama_kv_cache_context::set_input_relative_pos(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const {
    kv->set_input_relative_pos(dst, ubatch, extent);
}""", """void llama_kv_cache_context::set_input_relative_pos(ggml_tensor * dst, const llama_ubatch * ubatch, uint32_t extent) const {
    kv->set_input_relative_pos(dst, ubatch, extent);
}

void llama_kv_cache_context::set_input_relative_pos_rows(ggml_tensor * dst, const ggml_tensor * rows, const llama_ubatch * ubatch, uint32_t extent) const {
    kv->set_input_relative_pos_rows(dst, rows, ubatch, extent);
}""", n)
save(n, s)

# --- llama-graph.h: the input knows whether it uses the cell lists -------------------------------------------------
n = "llama-graph.h"
s = load(n)
s = rep(s, """    uint32_t n_kv_banded[2] = { 0, 0 };
    bool     banded         = false;
""", """    uint32_t n_kv_banded[2] = { 0, 0 };
    bool     banded         = false;

    // RAPIDLI: masks and distances over the kv_rows cell lists of each sequence (several sequences in the ubatch)
    bool     rows           = false;
    int64_t  n_head_rows    = 0;
""", n)
save(n, s)

# --- llama-graph.cpp -------------------------------------------------------------------------------------------------
n = "llama-graph.cpp"
s = load(n)
HELPER = """// RAPIDLI: Inkling's attention over each sequence's own cells (see engine/llama/patch_zzzrows.py), with kv_rows on and a
// per-head bias of at most 256M entries: a ubatch of several sequences (the lanes' passes), and a ubatch of one slice
// that the banded path cannot take (a prompt read whose cells are not in position order in a shared cache: dense, the
// bias spanned the whole cache - 32k of conversation context and 63 lanes, reads of 1,000 tokens took over 30 s)
static bool rel_use_kv_rows(const llama_memory_hybrid_iswa_context * mctx, const llama_ubatch & ubatch, const llama_cparams & cparams, int64_t n_head) {
    static const bool disabled = getenv("LLAMA_REL_ROWS_DISABLE") != nullptr;
    if (disabled || !cparams.fused_kv_rows || !cparams.flash_attn) {
        return false;
    }
    const auto * attn_ctx = mctx->get_attn();
    if (llama_kv_cache::get_n_kv_slices(ubatch) < 2 && ubatch.n_tokens > 32 &&
            (attn_ctx->get_base()->get_n_kv_pos_contiguous() > 0 || attn_ctx->get_swa()->get_n_kv_pos_contiguous() > 0)) {
        return false;                       // (the banded path, see llama_model_inkling2::graph::build_inputs)
    }
    const int64_t nb = attn_ctx->get_base()->get_n_kv_rows();
    const int64_t ns = attn_ctx->get_swa()->get_n_kv_rows();
    if (nb == 0 || ns == 0) {
        return false;
    }
    return std::max(nb, ns)*int64_t(ubatch.n_tokens)*std::max<int64_t>(1, n_head) <= (int64_t(1) << 28);
}

void llm_graph_input_mem_hybrid_iswa_rel::set_input(const llama_ubatch * ubatch) {"""
s = rep(s, "void llm_graph_input_mem_hybrid_iswa_rel::set_input(const llama_ubatch * ubatch) {", HELPER, n)
s = rep(s, """    if (self_rel_pos->buffer) {
        attn_ctx->get_base()->set_input_relative_pos(self_rel_pos, ubatch, extent);
    }""", """    if (self_rel_pos->buffer) {
        if (rows) {
            attn_ctx->get_base()->set_input_relative_pos_rows(self_rel_pos, inp_attn->self_attn_mask.rows, ubatch, extent);
        } else {
            attn_ctx->get_base()->set_input_relative_pos(self_rel_pos, ubatch, extent);
        }
    }""", n)
s = rep(s, """    if (self_rel_pos_swa->buffer) {
        attn_ctx->get_swa()->set_input_relative_pos(self_rel_pos_swa, ubatch, extent_swa);
    }""", """    if (self_rel_pos_swa->buffer) {
        if (rows) {
            attn_ctx->get_swa()->set_input_relative_pos_rows(self_rel_pos_swa, inp_attn->self_attn_mask_swa.rows, ubatch, extent_swa);
        } else {
            attn_ctx->get_swa()->set_input_relative_pos(self_rel_pos_swa, ubatch, extent_swa);
        }
    }""", n)
# can_reuse: the layout (cell lists or dense) must be the same, and so must the list lengths
s = rep(s, """    // the masks are always dense, see build_inp_mem_hybrid_iswa_rel()

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(inp_attn->self_attn_mask.raw, attn_ctx->get_base(), params.ubatch, params.cparams);

    res &= self_rel_pos->ne[0] == attn_ctx->get_base()->get_n_kv();
    res &= self_rel_pos->ne[1] == params.ubatch.n_tokens;

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    res &= can_reuse_kq_mask(inp_attn->self_attn_mask_swa.raw, attn_ctx->get_swa(), params.ubatch, params.cparams);

    res &= self_rel_pos_swa->ne[0] == attn_ctx->get_swa()->get_n_kv();
    res &= self_rel_pos_swa->ne[1] == params.ubatch.n_tokens;
""", """    // dense masks, or (RAPIDLI) masks over the kv_rows cell lists, see build_inp_mem_hybrid_iswa_rel()
    const bool use_rows = rel_use_kv_rows(mctx, params.ubatch, params.cparams, n_head_rows);
    if (use_rows != rows) {
        return false;
    }

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    if (rows) {
        res &= can_reuse_attn_mask(inp_attn->self_attn_mask, attn_ctx->get_base(), params.ubatch, params.cparams);
        res &= self_rel_pos->ne[0] == attn_ctx->get_base()->get_n_kv_rows();
    } else {
        res &= can_reuse_kq_mask(inp_attn->self_attn_mask.raw, attn_ctx->get_base(), params.ubatch, params.cparams);
        res &= self_rel_pos->ne[0] == attn_ctx->get_base()->get_n_kv();
    }
    res &= self_rel_pos->ne[1] == params.ubatch.n_tokens;

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
    }

    if (rows) {
        res &= can_reuse_attn_mask(inp_attn->self_attn_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);
        res &= self_rel_pos_swa->ne[0] == attn_ctx->get_swa()->get_n_kv_rows();
    } else {
        res &= can_reuse_kq_mask(inp_attn->self_attn_mask_swa.raw, attn_ctx->get_swa(), params.ubatch, params.cparams);
        res &= self_rel_pos_swa->ne[0] == attn_ctx->get_swa()->get_n_kv();
    }
    res &= self_rel_pos_swa->ne[1] == params.ubatch.n_tokens;
""", n)
# the builder: masks over the cell lists when the ubatch splits into sequences
s = rep(s, """    // dense masks: the bias is gathered by the distance to each cell of the K/V view
    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_attn_mask.raw = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_attn_mask_swa.raw = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
    }
""", """    // dense masks: the bias is gathered by the distance to each cell of the K/V view
    // (RAPIDLI: or over the kv_rows cell lists of each sequence, when the ubatch splits into sequences)
    const int64_t n_head_rows = hparams.n_head();
    const bool use_rows = rel_use_kv_rows(mctx_cur, ubatch, cparams, n_head_rows);
    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        if (use_rows) {
            inp_attn->self_attn_mask = build_attn_inp_attn_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        } else {
            inp_attn->self_attn_mask.raw = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        }
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        if (use_rows) {
            inp_attn->self_attn_mask_swa = build_attn_inp_attn_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        } else {
            inp_attn->self_attn_mask_swa.raw = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        }
    }
""", n)
s = rep(s, """    inp->self_rel_pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, attn_ctx->get_base()->get_n_kv(), ubatch.n_tokens);
    ggml_set_input(inp->self_rel_pos);
    ggml_set_name(inp->self_rel_pos, "attn_inp_rel_pos");

    inp->self_rel_pos_swa = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, attn_ctx->get_swa()->get_n_kv(), ubatch.n_tokens);""",
        """    inp->rows        = use_rows;
    inp->n_head_rows = n_head_rows;

    inp->self_rel_pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32,
            use_rows ? attn_ctx->get_base()->get_n_kv_rows() : attn_ctx->get_base()->get_n_kv(), ubatch.n_tokens);
    ggml_set_input(inp->self_rel_pos);
    ggml_set_name(inp->self_rel_pos, "attn_inp_rel_pos");

    inp->self_rel_pos_swa = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32,
            use_rows ? attn_ctx->get_swa()->get_n_kv_rows() : attn_ctx->get_swa()->get_n_kv(), ubatch.n_tokens);""", n)
# the attention: the cell lists go to the kv_rows flash attention with the bias in its per-head mask
s = rep(s, """    // TODO: support kv_rows - fill the distances over the kv_rows cell lists and pass the rows to build_attn_mha,
    //       needs a kv_rows flash attention with a per-head mask (CUDA has none)
    GGML_ASSERT(attn_mask.rows == nullptr);
""", """    // RAPIDLI: with kv_rows the distances were filled over the cell lists, and the per-head mask goes to the kv_rows
    // flash attention (ggml-cuda patched for a per-head mask with kv_rows: engine/cuda/patch_zrowsmask.py)
    GGML_ASSERT(attn_mask.rows == nullptr || cparams.flash_attn);
""", n)
s = rep(s, "    return build_attn_iswa_impl(*this, inp_attn, wo, wo_b, wo_s, q_cur, k_cur, v_cur, nullptr, kq_mask, nullptr, sinks, nullptr, kq_scale, il);",
        "    return build_attn_iswa_impl(*this, inp_attn, wo, wo_b, wo_s, q_cur, k_cur, v_cur, nullptr, kq_mask, attn_mask.rows, sinks, nullptr, kq_scale, il);", n)
save(n, s)

# --- models/inkling2.cpp: over the cell lists the bias is small, never split into row chunks ------------------------
n = os.path.join("models", "inkling2.cpp")
s = load(n)
s = rep(s, "            const int64_t n_kv_max = std::max<int64_t>(n_kv, (is_swa ? attn->mctx->get_swa() : attn->mctx->get_base())->get_size());",
        "            const int64_t n_kv_max = inp->rows ? n_kv : std::max<int64_t>(n_kv, (is_swa ? attn->mctx->get_swa() : attn->mctx->get_base())->get_size());", n)
save(n, s)
print("patched: llama-kv-cache.h/.cpp, llama-graph.h/.cpp, models/inkling2.cpp")
