@echo off
rem Builds rapidli-engine.exe (engine.cpp) against our llama.dll (..\llama) and installs it next to the libraries in ..\bin.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
if not exist obj mkdir obj
rem llama.cpp as Unsloth Studio installs it (headers, vendor code); set LLAMA_DIR to use another copy
if not defined LLAMA_DIR set LLAMA_DIR=%USERPROFILE%\.unsloth\llama.cpp
set U=%LLAMA_DIR%
set FLAGS=/nologo /c /O2 /MD /EHsc /std:c++17 /utf-8 /bigobj /DNDEBUG /DLLAMA_SHARED /DGGML_SHARED /D_CRT_SECURE_NO_WARNINGS /D_WIN32_WINNT=0x0A00 /DNOMINMAX /w /I%U%\include /I..\llama\ggml\include /I%U%\vendor /I%U%\tools\mtmd
if not exist obj\httplib.obj (
  echo compiling httplib
  cl %FLAGS% /Foobj\httplib.obj %U%\vendor\cpp-httplib\httplib.cpp > httplib.log 2>&1 || (echo HTTPLIB FAILED & exit /b 1)
)
echo compiling the engine
cl %FLAGS% /Foobj\engine.obj engine.cpp > engine.log 2>&1 || (echo ENGINE FAILED & exit /b 1)
echo linking
link /nologo /OUT:rapidli-engine.exe obj\engine.obj obj\httplib.obj ..\llama\bin\llama.lib ..\llama\lib\ggml.lib ..\llama\lib\ggml-base.lib ..\llama\lib\mtmd.lib ws2_32.lib > link.log 2>&1 || (echo LINK FAILED & exit /b 1)
copy /y rapidli-engine.exe ..\bin\rapidli-engine.exe >nul
echo ENGINE BUILD OK
