@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title parakeet-to-md :: install

rem Usage: install.bat [--cpu | --gpu]
rem   (nothing)  pick automatically: NVIDIA -> TensorRT/CUDA, AMD or Intel -> DirectML, else CPU
rem   --cpu     CPU only (smallest download, 416 MB)
rem   --gpu     force the GPU stack even if detection fails

set "FORCE=cpu"
if /i "%~1"=="--cpu" set "FORCE=cpu"
if /i "%~1"=="--gpu" set "FORCE=gpu"

set "UVDIR=%~dp0others\tools\uv"
set "UV="
set "VPY=others\.venv\Scripts\python.exe"
set "PY="

echo.
echo  parakeet-to-md :: install
echo.

for %%d in (input output archive others) do if not exist "%%d" mkdir "%%d" >nul 2>nul
for %%d in (input output archive) do if not exist "%%d\.gitkeep" echo.>"%%d\.gitkeep"

rem --- 1. python --------------------------------------------------------------
if exist "%VPY%" "%VPY%" -c "import sys" >nul 2>nul && goto :python_ok
if not defined PARAKEET_FORCE_UV (
  where python >nul 2>nul && for /f "tokens=2" %%v in ('python --version 2^>^&1') do (
    for /f "tokens=1,2 delims=." %%a in ("%%v") do if %%a GEQ 3 if %%b GEQ 10 set "PY=python"
  )
)
if defined PY (
  echo [1/5] python: %PY%
  goto :python_ok
)

echo [1/5] no usable Python 3.10+ - fetching a portable one ^(uv, no admin rights^)
if exist "%UVDIR%\uv.exe" goto :uv_ready
set "UVZIP=%TEMP%\parakeet-uv.zip"
del /q "%UVZIP%" >nul 2>nul
curl.exe -L --fail --silent --show-error --retry 3 -o "%UVZIP%" "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
if not exist "%UVZIP%" (
  echo [X] could not download uv. Check the network and rerun.
  goto :fail
)
if not exist "%UVDIR%" mkdir "%UVDIR%" >nul 2>nul
tar.exe -xf "%UVZIP%" -C "%UVDIR%" >nul 2>nul
del /q "%UVZIP%" >nul 2>nul
if not exist "%UVDIR%\uv.exe" (
  echo [X] uv.exe is not in others\tools\uv after unpacking.
  goto :fail
)
:uv_ready
set "UV=%UVDIR%\uv.exe"
"%UV%" python install 3.12
if errorlevel 1 (
  echo [X] could not install Python 3.12.
  goto :fail
)
echo [1/5] python: 3.12 ^(portable, managed by uv^)

:python_ok

rem --- 2. venv ----------------------------------------------------------------
if exist "%VPY%" goto :venv_ok
if not defined UV if exist "%UVDIR%\uv.exe" set "UV=%UVDIR%\uv.exe"
echo [2/5] creating others\.venv
if defined UV (
  "%UV%" venv "others\.venv" --python 3.12 --seed
) else (
  %PY% -m venv "others\.venv"
)
if errorlevel 1 (
  echo [X] could not create others\.venv - delete the folder and rerun.
  goto :fail
)
:venv_ok
if not defined UV if exist "%UVDIR%\uv.exe" set "UV=%UVDIR%\uv.exe"

rem --- 3. pick the GPU stack --------------------------------------------------
set "GPU=cpu"
if /i "%FORCE%"=="gpu" set "GPU=auto"
where nvidia-smi >nul 2>nul && set "GPU=nvidia"
if /i "%GPU%"=="cpu" (
  set "VID="
  for /f "delims=" %%d in ('powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController).Name -join ' '" 2^>nul') do set "VID=%%d"
  if defined VID echo !VID! | findstr /i "radeon amd arc intel" >nul && set "GPU=other"
)
if /i "%GPU%"=="auto" (
  echo [warn] --gpu was forced but no card was detected - installing the generic GPU stack.
  set "GPU=other"
)

if /i "%GPU%"=="nvidia" (
  echo [3/5] GPU: NVIDIA - TensorRT + CUDA 12 ^(pip wheels, no admin rights^)
  call :pipuninstall -y -q onnxruntime onnxruntime-directml onnxruntime-gpu
  call :pipinstall -e "others[gpu]" --quiet
  if errorlevel 1 (
    echo [warn] the NVIDIA stack did not install - continuing on CPU.
    set "GPU=cpu"
  )
)
if /i "%GPU%"=="other" (
  echo [3/5] GPU: AMD/Intel - DirectML ^(onnxruntime-directml^)
  call :pipuninstall -y -q onnxruntime onnxruntime-directml onnxruntime-gpu
  call :pipinstall -e "others" --quiet
  call :pipinstall "onnxruntime-directml>=1.20" --quiet
  if errorlevel 1 (
    echo [warn] DirectML did not install - continuing on CPU.
    set "GPU=cpu"
  )
)
if /i "%GPU%"=="cpu" (
  echo [3/5] CPU: onnxruntime ^(no usable GPU on this machine^)
)

rem --- 4. dependencies + verify ----------------------------------------------
if /i "%GPU%"=="cpu" (
  call :pipinstall -e "others[cpu]" --quiet
  if errorlevel 1 (
    echo [X] dependency install failed. Rerun with the full log:
    echo         %VPY% -m pip install -e others
    goto :fail
  )
)

echo [4/5] verifying that the engine loads
if /i "%GPU%"=="cpu" (set "DEV=cpu") else (set "DEV=gpu")
"%VPY%" -m parakeet_to_md --selftest --device %DEV%
if errorlevel 1 if /i not "%GPU%"=="cpu" (
  echo [warn] the GPU stack installed but no GPU session loaded - switching to CPU-only deps.
  call :pipuninstall -y -q onnxruntime-gpu tensorrt-cu12 tensorrt-cu12-libs onnxruntime-directml
  call :pipinstall "onnxruntime>=1.18" --quiet
  set "GPU=cpu"
)

rem --- 5. weights -------------------------------------------------------------
if /i "%GPU%"=="cpu" (set "SETS=cpu") else (set "SETS=both")
echo [5/5] downloading the weights ^(set: %SETS%^)
"%VPY%" -m parakeet_to_md.installer --set %SETS%
if errorlevel 1 (
  echo [X] model download failed - see the message above.
  goto :fail
)

echo.
echo running the self-check ...
if /i "%GPU%"=="cpu" (set "DEV=cpu") else (set "DEV=auto")
"%VPY%" -m parakeet_to_md --selftest --device %DEV%
if errorlevel 1 (
  echo.
  echo [X] self-check failed - the install is NOT usable yet. Read the FAIL lines above.
  goto :fail
)

echo.
echo  Install OK.
echo    input\    put your videos and audio here
echo    output\   one .md transcript per file
echo    archive\  sources already processed
echo.
echo  double-click run.bat to transcribe
echo  double-click uninstall.bat to remove
echo.
pause
exit /b 0

:pipinstall
if defined UV (
  "%UV%" pip install --python "%VPY%" %*
) else (
  "%VPY%" -m pip install %*
)
exit /b %errorlevel%

:pipuninstall
if defined UV (
  "%UV%" pip uninstall --python "%VPY%" %*
) else (
  "%VPY%" -m pip uninstall %*
)
exit /b %errorlevel%

:fail
echo.
pause
exit /b 2


