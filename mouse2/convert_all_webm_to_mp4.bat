@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo =========================================
echo   WEBM to MP4 Batch Converter
echo =========================================
echo Current folder:
echo %cd%
echo.

set "LOG=convert_log.txt"

echo WEBM to MP4 Batch Converter > "%LOG%"
echo Current folder: %cd% >> "%LOG%"
echo. >> "%LOG%"

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [ERROR] ffmpeg not found.
    echo [ERROR] ffmpeg not found. >> "%LOG%"
    echo.
    echo 请先确认 ffmpeg 已安装，并且可以在命令行运行：
    echo ffmpeg -version
    echo.
    pause
    exit /b 1
)

echo [OK] ffmpeg found.
echo [OK] ffmpeg found. >> "%LOG%"
echo.

set /a COUNT=0
set /a DONE=0
set /a SKIP=0
set /a FAIL=0

for /r "data\raw" %%f in (*.webm) do (
    set /a COUNT+=1
    set "input=%%f"
    set "output=%%~dpnf.mp4"

    echo -----------------------------------------
    echo Processing:
    echo %%f
    echo Processing: %%f >> "%LOG%"

    if exist "!output!" (
        echo [SKIP] MP4 already exists:
        echo !output!
        echo [SKIP] !output! >> "%LOG%"
        set /a SKIP+=1
    ) else (
        ffmpeg -y -i "%%f" -c:v libx264 -pix_fmt yuv420p -crf 18 -preset fast "!output!" >> "%LOG%" 2>&1

        if exist "!output!" (
            echo [DONE] Created:
            echo !output!
            echo [DONE] !output! >> "%LOG%"
            set /a DONE+=1
        ) else (
            echo [FAIL] Failed:
            echo %%f
            echo [FAIL] %%f >> "%LOG%"
            set /a FAIL+=1
        )
    )
)

echo.
echo =========================================
echo Finished.
echo Found WEBM: %COUNT%
echo Converted: %DONE%
echo Skipped: %SKIP%
echo Failed: %FAIL%
echo Log saved to: %LOG%
echo =========================================
echo.

echo Finished. >> "%LOG%"
echo Found WEBM: %COUNT% >> "%LOG%"
echo Converted: %DONE% >> "%LOG%"
echo Skipped: %SKIP% >> "%LOG%"
echo Failed: %FAIL% >> "%LOG%"

if %COUNT%==0 (
    echo [WARNING] 没有找到任何 .webm 文件。
    echo 请确认这个 bat 文件放在 mouse2 根目录。
    echo 例如：
    echo C:\Users\l2051\2022148103\mouse2\convert_all_webm_to_mp4.bat
    echo.
)

pause