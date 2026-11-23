@echo off
rem Builds ggml-cuda.dll, the CUDA backend, from the llama.cpp source in %LLAMA_SRC% (Unsloth's fork, commit c2cf5f9)
rem after the patches in this folder were applied to it (python patch_f16.py ... each once, on a fresh checkout).
rem Output: out\bin\ggml-cuda.dll; copy it into engine\bin.
rem   set LLAMA_SRC=D:\src\llama.cpp
rem   set CUDA_ARCH=89          (the card's compute capability: 89 = RTX 40, 86 = RTX 30, 75 = RTX 20; 120 = RTX 50; several: 75;86;89;120)
rem   build_cuda.bat configure
setlocal
if not defined LLAMA_SRC (echo set LLAMA_SRC to the llama.cpp source first & exit /b 1)
if not defined CUDA_ARCH set CUDA_ARCH=75;86;89;120
if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" >nul
rem CMake: on the PATH, else the copy that comes with Visual Studio (its "C++ CMake tools" component); Ninja: on the PATH
where cmake >nul 2>nul || set "PATH=%VSINSTALLDIR%Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VSINSTALLDIR%Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
where cmake >nul 2>nul || (echo CMake was not found: install it, or the "C++ CMake tools" of Visual Studio & exit /b 1)
where ninja >nul 2>nul || (echo Ninja was not found: install it ^(for example: winget install Ninja-build.Ninja^) & exit /b 1)
cd /d "%~dp0"
if "%1"=="configure" (
  rem (configured from the top of the source, with llama.cpp's own programs off: the published source has no ggml.pc.in,
  rem which a build of ggml alone needs)
  cmake -S "%LLAMA_SRC%" -B out -G Ninja -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON "-DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH%" -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
  if errorlevel 1 exit /b 1
)
cmake --build out --target ggml-cuda -j
if errorlevel 1 exit /b 1
echo CUDA BUILD OK
