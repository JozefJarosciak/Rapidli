Rapidli engine (built binaries)

Unpack this archive into the engine/bin folder of Rapidli (https://github.com/JozefJarosciak/Rapidli), so that
engine/bin/rapidli-engine.exe (Windows) or engine/bin/rapidli-engine (Linux) exists, then start Rapidli with
start.bat (Windows) or ./start.sh (Linux).

Windows (x64): needs an NVIDIA RTX 20 series card or newer with a current driver, and the CUDA Toolkit 13 (the engine
uses its cuBLAS libraries, found through CUDA_PATH). Native code for RTX 20, 30, 40 and 50 (compute 7.5, 8.6, 8.9,
12.0), built with CUDA 13.0.

Linux (x86-64): needs an NVIDIA RTX 20 series card or newer with its driver, and the CUDA 12 runtime libraries
(libcudart, libcublas: the CUDA Toolkit 12.x, in /usr/local/cuda/lib64 or on LD_LIBRARY_PATH). Built for RTX 20/30/40
with CUDA 12.6 under Ubuntu 22.04 (glibc 2.35 or newer). For an RTX 50 card, build the engine yourself with
engine/build-linux.sh and CUDA 12.8 or newer.

Rapidli has been tested on Windows 11 with an RTX 4090. The Linux engine was built and started under WSL 2 only; it has
not been tested on a Linux installed directly on a machine. Reports are welcome.

Built from Rapidli's source with llama.cpp (Unsloth's fork, release b12631-mix-acc0843, source 59f66204, MIT licence: LICENSE.llama.cpp and
THIRD_PARTY_LICENSES.txt in this archive). Rapidli itself is MIT licensed.
