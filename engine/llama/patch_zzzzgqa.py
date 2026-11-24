"""Patch for our copy of llama.cpp (src/), after patch_zzzrows.py: Inkling 2's lanes, one token each, attend with the
query heads that share a K/V head as the queries of one attention.

Over the kv_rows cell lists every lane is a slice with one token, so the flash attention ran one query per tile and
read the lane's cells (the shared prompt among them) once for each of the 24 query heads: Inkling-2-Mini has 3 query
heads per K/V head, a ratio the kernel cannot group (it groups 2, 4 or 8, and not at all with a per-head mask). Here
the 3 heads of a K/V head are the 3 queries of an 8-head attention: each K/V head's cells are read once for the three,
and a tile holds 3 queries instead of 1. The per-head bias mask has the same bytes in both shapes
([rows, 1, 24, lanes] = [rows, 3, 8, lanes]); the output is put back in head order. Only for ubatches of one token per
slice with grouped heads. LLAMA_REL_ROWS_NOGROUP=1: as before.
  python patch_zzzzgqa.py"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
NL = chr(10)
n = os.path.join("models", "inkling2.cpp")
p = os.path.join(SRC, n)
s = open(p, encoding="utf8").read().replace(chr(13) + NL, NL)
if "RAPIDLI-GQA" in s:
    raise SystemExit("already patched")
a = """        } else {
            // project before gathering: only the gathered bias is n_kv x n_tokens x n_head"""
b = """        } else if (inp->rows && n_head > n_head_kv && n_head % n_head_kv == 0 &&
                n_tokens == (is_swa ? attn->self_attn_mask_swa.rows : attn->self_attn_mask.rows)->ne[1] && !rows_nogroup()) {
            // RAPIDLI-GQA: one token per lane over the kv_rows lists: the g query heads of a K/V head are the g queries of
            // an n_head_kv-head attention (see engine/llama/patch_zzzzgqa.py)
            const auto & am = is_swa ? attn->self_attn_mask_swa : attn->self_attn_mask;
            const int64_t n_slices = am.rows->ne[1];
            const int64_t g        = n_head / n_head_kv;
            const int64_t n_rows   = rel_pos->ne[0];

            // the dense path's bias (minus the token's sink) over the cell lists: [n_rows, n_head, n_tokens]
            ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);
            table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens));
            table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));
            ggml_tensor * kq_b = ggml_get_rows(ctx0, table, rel_pos);
            kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 1, 0, 2, 3));
            cb(kq_b, "kq_b", il);
            ggml_tensor *& mdev = inp->self_kq_mask_dev[is_swa];
            if (!mdev) {
                mdev = ggml_cast(ctx0, am.raw, GGML_TYPE_F32);
            }
            ggml_tensor * kq_mask = ggml_add(ctx0, kq_b, ggml_reshape_3d(ctx0, mdev, n_rows, 1, n_slices));
            kq_mask = ggml_reshape_4d(ctx0, ggml_cast(ctx0, kq_mask, GGML_TYPE_F16), n_rows, g, n_head_kv, n_slices);

            const auto * cache = is_swa ? attn->mctx->get_swa() : attn->mctx->get_base();
            auto * k_rot = is_swa ? attn->self_k_rot_swa[il] : attn->self_k_rot[il];
            auto * v_rot = is_swa ? attn->self_v_rot_swa[il] : attn->self_v_rot[il];
            if (k_rot) {
                Qcur = llama_mul_mat_hadamard(ctx0, Qcur, k_rot);
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
            }
            if (v_rot) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
            }
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);
            ggml_build_forward_expand(gf, cache->cpy_k(ctx0, Kcur, is_swa ? attn->get_k_idxs_swa() : attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, cache->cpy_v(ctx0, Vcur, is_swa ? attn->get_v_idxs_swa() : attn->get_v_idxs(), il));

            ggml_tensor * k_all = cache->get_k(ctx0, il);
            ggml_tensor * v_all = cache->get_v(ctx0, il);
            GGML_ASSERT(!(v_all->nb[1] > v_all->nb[2]) && "grouped heads need the flash-attention V layout");
            // query head kv*g + j is query j of head kv
            ggml_tensor * q_fa = ggml_reshape_4d(ctx0, ggml_cont(ctx0, Qcur), n_embd_head, g, n_head_kv, n_slices);
            ggml_tensor * k_fa = ggml_permute(ctx0, k_all, 0, 2, 1, 3);
            ggml_tensor * v_fa = ggml_permute(ctx0, v_all, 0, 2, 1, 3);

            ggml_tensor *& zero = clip_limits[{ { model.dev_layer(il), -n_head_kv }, 0.0f }];
            if (!zero) {
                zero = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head_kv), 0.0f);
            }
            cur = ggml_flash_attn_ext_rows(ctx0, q_fa, k_fa, v_fa, kq_mask, am.rows, 1.0f/sqrtf(n_embd_head), 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(cur, zero);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN_KV_ROWS, cur, il});
            // [hd, n_head_kv, g, lanes] -> head order [hd, n_head, n_tokens]
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
            cur = ggml_reshape_2d(ctx0, cur, n_embd_head*n_head, n_tokens);
            cb(cur, "kqv_out", il);
            if (v_rot) {
                cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
            }
        } else {
            // project before gathering: only the gathered bias is n_kv x n_tokens x n_head"""
assert s.count(a) == 1, s.count(a)
s = s.replace(a, b)
# the switch, once per process
a2 = "ggml_tensor * llama_model_inkling2::graph::build_layer("
assert s.count(a2) == 1, s.count(a2)
s = s.replace(a2, """// RAPIDLI-GQA: LLAMA_REL_ROWS_NOGROUP=1 turns the grouped heads off (see engine/llama/patch_zzzzgqa.py)
static bool rows_nogroup() {
    static const bool off = getenv("LLAMA_REL_ROWS_NOGROUP") != nullptr;
    return off;
}

""" + a2)
open(p, "w", encoding="utf8", newline=NL).write(s)
print("patched", n)
