@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title parakeet-to-md :: uninstall

echo ============================================================
echo  parakeet-to-md :: uninstall
echo.
echo  Default: removes what install.bat downloaded and built
echo    others\   program, Python environment, models, ffmpeg   (~2 GB)
echo  and KEEPS your work
echo    input\  output\  archive\
echo.
echo  uninstall.bat --purge   also deletes input\, output\ and
echo                          archive\ - your videos, transcripts
echo                          and archives. Irreversible; asks you
echo                          to type YES.
echo ============================================================
echo.

set "PURGE="
if /i "%~1"=="--purge" set "PURGE=1"

if defined PURGE goto :confirm_purge

set "ANS="
set /p "ANS=Remove the program and keep your transcripts? [y/N] > "
if /i not "%ANS%"=="y" (
  echo.
  echo Cancelled. Nothing was deleted.
  pause
  exit /b 1
)
goto :do_remove

:confirm_purge
echo [^!] --purge also deletes input\, output\ and archive\:
echo     every video, every audio file and every transcript.
echo.
set "ANS="
set /p "ANS=Type YES (all caps) to delete everything, anything else cancels > "
if /i not "%ANS%"=="YES" (
  echo.
  echo Cancelled. Nothing was deleted.
  pause
  exit /b 1
)

:do_remove
echo.
echo removing others\ ...
if exist "others" rmdir /s /q "others"
rem Flat layout, for a copy installed before 2026-10-27. Harmless on a new install.
for %%d in (app models tools .venv) do if exist "%%d" rmdir /s /q "%%d"
for %%f in (pyproject.toml requirements.txt README.md LICENSE) do if exist "%%f" del /q "%%f" >nul 2>nul

if exist "others" (
  echo.
  echo [X] others\ is still there - something is using it. Close any parakeet-to-md
  echo     window or editor open in this folder and run me again.
  pause
  exit /b 2
)

if defined PURGE (
  echo removing input\, output\, archive\ ...
  for %%d in (input output archive) do if exist "%%d" rmdir /s /q "%%d"
)

echo.
echo ============================================================
if defined PURGE (
  echo Everything is gone except these .bat files. Delete this folder
  echo itself to remove them too.
) else (
  echo Uninstalled. Your work is untouched:
  echo    output\    your transcripts
  echo    archive\   the files already processed
  echo    input\     anything that was not processed yet
  echo.
  echo To reinstall: double-click install.bat - it downloads again.
  echo To wipe everything: uninstall.bat --purge
)
echo ============================================================
pause
exit /b 0
