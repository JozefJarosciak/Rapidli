"""Patch for llama.dll: the MoE cache's host experts are registered as pinned memory (cudaHostRegister, read only).

The cache uploads the experts it misses from the mapped model file. From pageable pages the driver copies through its
own staging buffers: 9.8 GB/s on an RTX 4090 (PCIe 4). The same pages registered as pinned copy at 23.5 GB/s
(measured 22 Nov 2026). With Qwen3.8-Flash-Next on 30 lanes, a long article misses 12-15% of its experts
and uploads about 1.4 GB per pass, so the uploads took most of each pass.

The expert tensors of all cached layers are merged into ranges (gaps under 16 MiB joined, so no page is registered
twice) and each range is registered once per process; both contexts' caches share them. A range that cannot be locked
(not enough free memory) stays pageable, as before. LLAMA_MOE_PIN=0 turns it off.
  python patch_zzmoepin.py"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))


def edit(name, pairs=()):
    p = os.path.join(HERE, "src", name)
    s = open(p, encoding="utf8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, a[:60], s.count(a))
        s = s.replace(a, b)
    open(p, "w", encoding="utf8", newline="\n").write(s)


edit("llama-moe-cache.cpp", [
    ("#include <cstring>\n", "#include <cstring>\n#include <mutex>    // patch_zzmoepin.py\n"),
    ("""                        (double) g.n_slots/(g.layers.size()*n_expert_used));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {""",
     """                        (double) g.n_slots/(g.layers.size()*n_expert_used));
                }
            }
        }
        if (!no_alloc) {
            pin_host_experts();
        }
    }

    // (patch_zzmoepin.py) the host experts as pinned memory: uploads of misses at ~23 instead of ~10 GB/s
    void pin_host_experts() {
        static std::mutex mtx;
        static bool done = false;
        std::lock_guard<std::mutex> lock(mtx);
        const char * env = getenv("LLAMA_MOE_PIN");
        if (done || (env != nullptr && atoi(env) == 0) || devices.empty()) {
            return;
        }
        done = true;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(devices[0].backend));
        auto reg_fn = reg ? (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_host_buffer") : nullptr;
        if (reg_fn == nullptr) {
            return;
        }
#if defined(_WIN32)
        if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) { _putenv_s("GGML_CUDA_REGISTER_HOST", "1"); }
#else
        setenv("GGML_CUDA_REGISTER_HOST", "1", 0);
#endif
        std::vector<std::pair<uintptr_t, uintptr_t>> r;
        for (const group & g : groups) {
            for (int32_t il : g.layers) {
                for (const ggml_tensor * t : layers[il].experts) {
                    if (t->data != nullptr) {
                        r.push_back({ (uintptr_t) t->data, (uintptr_t) t->data + ggml_nbytes(t) });
                    }
                }
            }
        }
        std::sort(r.begin(), r.end());
        std::vector<std::pair<uintptr_t, uintptr_t>> m;
        for (const auto & x : r) {
            if (!m.empty() && x.first <= m.back().second + (16u << 20)) {
                m.back().second = std::max(m.back().second, x.second);
            } else {
                m.push_back(x);
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        size_t ok = 0, all = 0;
        for (const auto & x : m) {
            all += x.second - x.first;
            if (reg_fn((void *) x.first, x.second - x.first)) {
                ok += x.second - x.first;
            }
        }
        LLAMA_LOG_INFO("llama_moe_cache: %.2f of %.2f GiB of host experts pinned in %zu ranges (%.1f s)\\n",
            ok/1073741824.0, all/1073741824.0, m.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    ggml_backend_t backend(int32_t il) const {"""),
])
