@echo off
:: ============================================================
::  Video Upscale Pipeline Launcher
::  Drop movie into C:\VideoUpscale\input\  then run this
:: ============================================================
setlocal
set BASE=C:\VideoUpscale
set PYTHON=%BASE%\venv\Scripts\python.exe

echo.
echo  ============================================
echo   AI Video Upscale Pipeline  (RTX 3070)
echo  ============================================
echo.
echo  Pipeline (RTX 3070 8GB optimised):
echo    Deblock ^> Denoise ^> [Stabilize] ^> HAT 4x (30fps tiles)
echo    ^> RIFE 2x FPS at 4K ^> Sharpen ^> Colour Correct ^> x265
echo.

for %%F in ("%BASE%\input\*.mkv" "%BASE%\input\*.mp4" "%BASE%\input\*.avi" "%BASE%\input\*.mov" "%BASE%\input\*.wmv") do (
    if not defined INPUT_FILE set INPUT_FILE=%%F
)

if not defined INPUT_FILE (
    echo  ERROR: No video found in %BASE%\input\
    echo  Supported formats: mkv mp4 avi mov wmv
    pause & exit /b 1
)

echo  Found: %INPUT_FILE%
echo.

echo  Denoise strength:
echo    [1] Low     (preserve more detail, less smoothing)
echo    [2] Medium  (recommended for most bad-quality sources)
echo.
set /p DCHOICE="Enter 1 or 2 (default=2): "
if "%DCHOICE%"=="1" (set DENOISE=low) else (set DENOISE=medium)

echo.
echo  Stabilize shaky camera?
echo    [Y] Yes  (adds extra pass, good for handheld/shaky footage)
echo    [N] No   (skip, faster)
echo.
set /p STAB="Stabilize? Y/N (default=N): "
set STAB_FLAG=
if /i "%STAB%"=="Y" set STAB_FLAG=--stabilize

echo.
echo  ============================================
echo   Starting...
echo   Denoise  : %DENOISE%
echo   Stabilize: %STAB%
echo   Upscale  : 4x  (Real-ESRGAN CUDA)
echo   FPS      : 2x  (RIFE v4.6)
echo   Encode   : x265 CRF16
echo  ============================================
echo.

"%PYTHON%" "%BASE%\upscale.py" ^
    --input "%INPUT_FILE%" ^
    --scale 4 ^
    --tile 384 ^
    --model Real_HAT_GAN_SRx4_sharper ^
    --rife-order late ^
    --denoise %DENOISE% ^
    %STAB_FLAG%

echo.
echo  Output saved to: %BASE%\output\
pause
