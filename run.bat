@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title parakeet-to-md :: run

rem Pause only when double-clicked, so `run.bat --only foo.mp4` from a terminal returns normally.
set "PAUSE_AT_END="
echo %cmdcmdline% | find /i "%~nx0" >nul && set "PAUSE_AT_END=1"

rem The venv lives in others\ so the user's folder shows four entries: input, output, archive, others.
rem A flat install made before 2026-10-27 still has it at .venv, and is accepted here so an old copy
rem keeps running until install.bat upgrades it.
set "VPY=others\.venv\Scripts\python.exe"
if not exist "%VPY%" if exist ".venv\Scripts\python.exe" set "VPY=.venv\Scripts\python.exe"

if not exist "%VPY%" (
  echo [X] not installed yet - double-click install.bat first.
  if defined PAUSE_AT_END pause
  exit /b 2
)

"%VPY%" -m parakeet_to_md %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo Finished. Transcripts are in output\ , processed sources in archive\ .
) else (
  echo Finished WITH ERRORS ^(exit %RC%^). Failed files were left in input\ .
  echo The lines above name the file and the reason.
)

if defined PAUSE_AT_END pause
exit /b %RC%
