@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: ============================================================
::  run_system.bat
::  Automated Synthetic Data Generator ^& Model Alignment Pipeline
::  Single-click initialization script with 3-stage pre-flight
:: ============================================================
::
::  STAGE 1 -- LOCAL ENVIRONMENT DIAGNOSTIC
::  STAGE 2 -- CONFIGURATION SCHEMA PRE-FLIGHT  (Pydantic / .env)
::  STAGE 3 -- AUTOMATED APP DISPATCH           (Streamlit :8502)
::
::  Exit codes:
::    0  -- Clean launch
::    1  -- Virtual environment missing
::    2  -- Python interpreter missing inside venv
::    3  -- Environment / API key validation failed
:: ============================================================

:: ── Capture script directory BEFORE CD so %~dp0 is always safe.
::    SET "var=value" form (quotes wrap the whole assignment) is the
::    only Windows batch idiom that stores paths containing & correctly.
::    The value stored in SCRIPT_DIR does NOT include the surrounding
::    quotes; they are syntax, not content.
SET "SCRIPT_DIR=%~dp0"

:: Strip the trailing backslash that %~dp0 appends (cosmetic only).
IF "!SCRIPT_DIR:~-1!" == "\" SET "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"

:: Change to the project root (quoted for spaces + special chars).
CD /D "!SCRIPT_DIR!"

:: ── Use a relative PYTHONPATH: "." always maps to wherever we CD'd. ──
::    This avoids re-expanding SCRIPT_DIR (with its embedded &) through
::    any command that would misparse it as a command separator.
SET "PYTHONPATH=."

:: ── Default terminal colour: bright white on black ──
COLOR 0F
CLS

:: ============================================================
::  BANNER
:: ============================================================
echo.
echo  ================================================================
echo   SYNTHETIC DATA GENERATOR ^& MODEL ALIGNMENT PIPELINE
echo   System Initialization Script  ^|  run_system.bat
echo  ================================================================
echo.
echo  Working directory : "!SCRIPT_DIR!"
echo  Timestamp         : %DATE%  %TIME%
echo.

:: ============================================================
::  STAGE 1 -- LOCAL ENVIRONMENT DIAGNOSTIC
:: ============================================================
COLOR 03
echo  ----------------------------------------------------------------
echo  [STAGE 1]  LOCAL ENVIRONMENT DIAGNOSTIC
echo  ----------------------------------------------------------------
COLOR 0F
echo.

:: ── 1a. Virtual environment activation script ──
echo  [1/3]  Checking virtual environment ...
IF NOT EXIST ".venv\Scripts\activate.bat" (
    COLOR 0C
    echo.
    echo  ================================================================
    echo   FATAL ERROR -- Virtual environment not found
    echo  ================================================================
    echo.
    echo   Expected path:
    echo     "!SCRIPT_DIR!\.venv\Scripts\activate.bat"
    echo.
    echo   Resolution -- run from the project root, then retry:
    echo.
    echo     py -3.12 -m venv .venv
    echo     .venv\Scripts\activate.bat
    echo     pip install -r requirements.txt
    echo.
    echo  ================================================================
    COLOR 0F
    echo.
    pause
    EXIT /B 1
)
echo       [OK]  .venv\Scripts\activate.bat found.

:: ── 1b. Python interpreter inside venv ──
echo  [2/3]  Checking venv Python interpreter ...
IF NOT EXIST ".venv\Scripts\python.exe" (
    COLOR 0C
    echo.
    echo  ================================================================
    echo   FATAL ERROR -- Python interpreter missing inside venv
    echo  ================================================================
    echo.
    echo   Expected path:
    echo     "!SCRIPT_DIR!\.venv\Scripts\python.exe"
    echo.
    echo   The virtual environment appears incomplete.
    echo   Resolution -- delete and recreate the venv:
    echo.
    echo     rmdir /S /Q .venv
    echo     py -3.12 -m venv .venv
    echo     .venv\Scripts\activate.bat
    echo     pip install -r requirements.txt
    echo.
    echo  ================================================================
    COLOR 0F
    echo.
    pause
    EXIT /B 2
)
echo       [OK]  .venv\Scripts\python.exe confirmed.

:: ── 1c. Python version must be 3.12 ──
echo  [3/3]  Verifying Python version constraint (3.12) ...
FOR /F "tokens=2 delims= " %%V IN ('".venv\Scripts\python.exe" --version 2^>^&1') DO SET "PY_VER=%%V"
echo       Interpreter reports: Python !PY_VER!
echo !PY_VER! | findstr /B "3.12" >NUL 2>&1
IF %ERRORLEVEL% NEQ 0 (
    COLOR 0E
    echo.
    echo  ================================================================
    echo   WARNING -- Python version mismatch
    echo  ================================================================
    echo.
    echo   This project targets Python 3.12.
    echo   Detected version : !PY_VER!
    echo.
    echo   Execution will continue, but unexpected behaviour may occur.
    echo   To realign, recreate the venv: py -3.12 -m venv .venv
    echo.
    echo  ================================================================
    COLOR 0F
    echo.
    echo   Press any key to continue anyway, or Ctrl+C to abort ...
    pause >NUL
)

COLOR 0A
echo.
echo   STAGE 1 PASSED -- Environment structure is valid.
COLOR 0F
echo.

:: ============================================================
::  STAGE 2 -- CONFIGURATION SCHEMA PRE-FLIGHT
:: ============================================================
COLOR 03
echo  ----------------------------------------------------------------
echo  [STAGE 2]  CONFIGURATION SCHEMA PRE-FLIGHT
echo  ----------------------------------------------------------------
COLOR 0F
echo.
echo   Validating Pydantic settings and .env context ...
echo   Executing: .venv\Scripts\python.exe config/settings.py
echo.

:: PYTHONPATH is already set to "." (relative), so no & exposure here.
.venv\Scripts\python.exe config/settings.py

IF %ERRORLEVEL% NEQ 0 (
    COLOR 0C
    echo.
    echo  ================================================================
    echo   FATAL ERROR -- Environment / API key validation failed
    echo  ================================================================
    echo.
    echo   Pydantic settings validation exited with a non-zero error code.
    echo   The process has been aborted to prevent running a broken state.
    echo.
    echo   Most likely causes:
    echo     1. .env file is missing from the project root.
    echo     2. GEMINI_API_KEY is not set or contains only whitespace.
    echo     3. OUTPUT_DIR value is invalid or cannot be created.
    echo.
    echo   Resolution:
    echo     1. Copy the template : copy .env.example .env
    echo     2. Edit .env and set : GEMINI_API_KEY=^<your-google-ai-studio-key^>
    echo     3. Re-run this script.
    echo.
    echo   Obtain an API key at: https://aistudio.google.com/app/apikey
    echo.
    echo  ================================================================
    COLOR 0F
    echo.
    pause
    EXIT /B 3
)

COLOR 0A
echo.
echo   STAGE 2 PASSED -- All environment variables validated successfully.
COLOR 0F
echo.

:: ============================================================
::  STAGE 3 -- AUTOMATED APP DISPATCH
:: ============================================================
COLOR 03
echo  ----------------------------------------------------------------
echo  [STAGE 3]  AUTOMATED APP DISPATCH
echo  ----------------------------------------------------------------
COLOR 0F
echo.
echo   All pre-flight checks passed. Launching Streamlit dashboard ...
echo.
echo   +---------------------------------------------------------+
echo   ^|  Dashboard URL:  http://localhost:8502                  ^|
echo   ^|  Press Ctrl+C in this terminal to stop the server.     ^|
echo   +---------------------------------------------------------+
echo.

COLOR 0A
echo   Starting server on port 8502 ...
COLOR 0F
echo.

:: ── Single unbroken command line — no line-continuation carets.
::    Continuation carets (^) at end-of-line can silently consume the
::    next line when the path or working directory contains special chars.
.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.port 8502 --server.headless false --browser.gatherUsageStats false

:: ── Post-run handler ─────────────────────────────────────────
IF %ERRORLEVEL% NEQ 0 (
    COLOR 0C
    echo.
    echo  ================================================================
    echo   Streamlit exited with an unexpected error
    echo  ================================================================
    echo.
    echo   Check the stack trace above for details.
    echo   Common causes:
    echo     - Port 8502 already in use -- kill the existing process first.
    echo     - Import error in dashboard/app.py or a pipeline module.
    echo     - Missing dependency -- re-run: pip install -r requirements.txt
    echo.
    echo  ================================================================
    COLOR 0F
    echo.
)

pause
ENDLOCAL
