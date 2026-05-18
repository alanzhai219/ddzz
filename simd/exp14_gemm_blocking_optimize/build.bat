@echo off
setlocal

where cl >nul 2>nul
if errorlevel 1 (
    echo cl compiler not found. Please run this script from a Visual Studio Developer Command Prompt.
    exit /b 1
)

if exist gemm.obj del /q gemm.obj

cl /nologo /std:c++20 /O2 /arch:AVX2 /EHsc /fp:fast gemm.cpp /Fe:gemm.exe
if errorlevel 1 exit /b %errorlevel%

if exist gemm.obj del /q gemm.obj

endlocal
