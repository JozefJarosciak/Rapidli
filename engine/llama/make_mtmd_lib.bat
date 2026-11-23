@echo off
rem Import library for mtmd.dll (llama.cpp's multimodal library, from Unsloth's build): the engine links it for images.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
dumpbin /exports ..\bin\mtmd.dll > lib\mtmd.exports.txt || (echo DUMPBIN FAILED & exit /b 1)
py make_def.py mtmd || (echo DEF FAILED & exit /b 1)
lib /nologo /def:lib\mtmd.def /out:lib\mtmd.lib /machine:x64 > lib\mtmd.lib.log 2>&1 || (echo LIB FAILED & exit /b 1)
echo MTMD LIB OK
