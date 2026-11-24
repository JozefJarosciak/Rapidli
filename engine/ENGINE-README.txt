Rapidli engine (built binaries)

Unpack this archive into the engine/bin folder of Rapidli (https://github.com/JozefJarosciak/Rapidli), so that
engine/bin/rapidli-engine.exe (Windows) or engine/bin/rapidli-engine (Linux) exists, then start Rapidli with
start.bat (Windows) or ./start.sh (Linux).

The CUDA libraries the engine uses come with it, so only the NVIDIA driver has to be installed (no CUDA Toolkit).

Windows (x64): needs an NVIDIA RTX 20 series card or newer with driver 580 or newer (CUDA 13). cuBLAS
(cublas64_13.dll, cublasLt64_13.dll, CUDA 13.0) is in this archive. Native code for RTX 20, 30, 40 and 50 (compute
7.5, 8.6, 8.9, 12.0), built with CUDA 13.0.

Linux (x86-64): needs an NVIDIA RTX 20 series card or newer with driver 525 or newer. The CUDA 12.6 runtime and cuBLAS
(libcudart.so.12, libcublas.so.12, libcublasLt.so.12) are in this archive; Rapidli puts engine/bin first on the
engine's LD_LIBRARY_PATH. Built for RTX 20/30/40 with CUDA 12.6 under Ubuntu 22.04 (glibc 2.35 or newer). For an
RTX 50 card, build the engine yourself with engine/build-linux.sh and CUDA 12.8 or newer.

Rapidli has been tested on Windows 11 with an RTX 4090. The Linux engine was built and started under WSL 2 only; it has
not been tested on a Linux installed directly on a machine. Reports are welcome.

Built from Rapidli's source with llama.cpp (Unsloth's fork, release b12631-mix-acc0843, source 59f66204, MIT licence: LICENSE.llama.cpp and
THIRD_PARTY_LICENSES.txt in this archive). Rapidli itself is MIT licensed. The NVIDIA CUDA libraries are NVIDIA's,
redistributed under the CUDA EULA (LICENSE.cuda.txt in this archive).
