@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
rem (a fresh clone has none of these folders)
for %%d in (lib bin obj\core obj\models) do if not exist %%d mkdir %%d
if not defined LLAMA_DIR set LLAMA_DIR=%USERPROFILE%\.unsloth\llama.cpp
set BIN=%LLAMA_DIR%\build\bin\Release
dumpbin /nologo /exports "%BIN%\ggml.dll" > lib\ggml.exports.txt
dumpbin /nologo /exports "%BIN%\ggml-base.dll" > lib\ggml-base.exports.txt
py make_def.py ggml || exit /b 1
py make_def.py ggml-base || exit /b 1
lib /nologo /def:lib\ggml.def /machine:x64 /out:lib\ggml.lib || exit /b 1
lib /nologo /def:lib\ggml-base.def /machine:x64 /out:lib\ggml-base.lib || exit /b 1
set FLAGS=/nologo /c /MP /O2 /MD /EHsc /std:c++17 /utf-8 /bigobj /DNDEBUG /DLLAMA_BUILD /DLLAMA_SHARED /DGGML_SHARED /D_CRT_SECURE_NO_WARNINGS /w /Isrc /Iinclude /Iggml\include /Igen
echo compiling core
cl %FLAGS% /Foobj\core\ src\*.cpp > core.log 2>&1 || (echo CORE FAILED & exit /b 1)
echo compiling models
cl %FLAGS% /Foobj\models\ src\models\*.cpp > models.log 2>&1 || (echo MODELS FAILED & exit /b 1)
echo linking
link /nologo /DLL /OUT:bin\llama.dll obj\core\*.obj obj\models\*.obj lib\ggml.lib lib\ggml-base.lib > link.log 2>&1 || (echo LINK FAILED & exit /b 1)
echo BUILD OK
