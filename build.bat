@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem build.bat
rem Windows build script for sdanalysis_kun using uv + PyInstaller.
rem
rem Usage:
rem   build.bat
rem
rem Optional overrides:
rem   set APP_NAME=SDAnalysis-kun
rem   set PACKAGE_NAME=sdanalysis_kun
rem   set ENTRY_SCRIPT=.\src\sdanalysis_kun\cli.py
rem   set ICON_FILE=.\src\sdanalysis_kun\img\icon.ico
rem   build.bat

if "%APP_NAME%"=="" set "APP_NAME=SDAnalysis-kun"
if "%PACKAGE_NAME%"=="" set "PACKAGE_NAME=sdanalysis_kun"
if "%SRC_PATH%"=="" set "SRC_PATH=.\src"
if "%ENTRY_SCRIPT%"=="" set "ENTRY_SCRIPT=.\src\sdanalysis_kun\cli.py"
if "%ICON_FILE%"=="" set "ICON_FILE=.\src\sdanalysis_kun\img\icon.ico"
if "%VERSION_FILE%"=="" set "VERSION_FILE=.\app.version"

where uv >nul 2>nul
if errorlevel 1 (
  echo Error: uv is not installed or not in PATH.
  echo Install example: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
  exit /b 1
)

if not exist "%ENTRY_SCRIPT%" (
  if exist ".\scripts\launch_app.py" set "ENTRY_SCRIPT=.\scripts\launch_app.py"
)
if not exist "%ENTRY_SCRIPT%" (
  if exist ".\scripts\launch.py" set "ENTRY_SCRIPT=.\scripts\launch.py"
)
if not exist "%ENTRY_SCRIPT%" (
  if exist ".\main.py" set "ENTRY_SCRIPT=.\main.py"
)
if not exist "%ENTRY_SCRIPT%" (
  if exist ".\src\%PACKAGE_NAME%\cli.py" set "ENTRY_SCRIPT=.\src\%PACKAGE_NAME%\cli.py"
)
if not exist "%ENTRY_SCRIPT%" (
  echo Error: entry script not found.
  echo Tried: .\src\sdanalysis_kun\cli.py, .\scripts\launch_app.py, .\scripts\launch.py, .\main.py, .\src\%PACKAGE_NAME%\cli.py
  echo Set ENTRY_SCRIPT to the actual launcher path and run again.
  exit /b 1
)

if not exist "%VERSION_FILE%" (
  echo Error: version file not found: %VERSION_FILE%
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo ==^> Creating uv virtual environment
  uv venv
  if errorlevel 1 exit /b 1
)

echo ==^> Installing project and build dependencies
uv pip install -e . pyinstaller pyinstaller-versionfile
if errorlevel 1 exit /b 1

echo ==^> Building Windows exe with PyInstaller
if exist "%ICON_FILE%" (
  uv run pyinstaller --onefile --windowed --clean --path "%SRC_PATH%" --collect-all "%PACKAGE_NAME%" -n "%APP_NAME%" --icon="%ICON_FILE%" --version-file="%VERSION_FILE%" "%ENTRY_SCRIPT%"
) else (
  echo Warning: icon file not found: %ICON_FILE%
  echo Building without a custom icon. Add icon.ico later and rerun for release builds.
  uv run pyinstaller --onefile --windowed --clean --path "%SRC_PATH%" --collect-all "%PACKAGE_NAME%" -n "%APP_NAME%" --version-file="%VERSION_FILE%" "%ENTRY_SCRIPT%"
)
if errorlevel 1 exit /b 1

echo ==^> Done: .\dist\%APP_NAME%.exe
endlocal
