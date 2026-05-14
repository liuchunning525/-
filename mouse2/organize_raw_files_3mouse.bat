@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo =========================================
echo   Organizing raw files by mouse/grip/condition
echo =========================================

for /d %%u in (data\raw\user_*) do (
    echo.
    echo User folder: %%u

    for %%f in ("%%u\*.webm" "%%u\*.json" "%%u\*.mp4") do (
        if exist "%%~f" (
            set "filename=%%~nxf"
            set "name=%%~nf"

            set "mouse=unknown"
            set "grip=unknown"
            set "cond=unknown"

            echo !filename! | findstr /i "G102" >nul && set "mouse=G102"
            echo !filename! | findstr /i "XliteV3ES" >nul && set "mouse=XliteV3ES"
            echo !filename! | findstr /i "X2H" >nul && set "mouse=X2H"

            echo !filename! | findstr /i "_palm_" >nul && set "grip=palm"
            echo !filename! | findstr /i "_claw_" >nul && set "grip=claw"
            echo !filename! | findstr /i "_fingertip_" >nul && set "grip=fingertip"

            echo !filename! | findstr /i "_A_" >nul && set "cond=Condition A"
            echo !filename! | findstr /i "_B_" >nul && set "cond=Condition B"

            if "!mouse!"=="unknown" (
                echo [SKIP] Cannot infer mouse: !filename!
            ) else if "!grip!"=="unknown" (
                echo [SKIP] Cannot infer grip: !filename!
            ) else if "!cond!"=="unknown" (
                echo [SKIP] Cannot infer condition: !filename!
            ) else (
                set "target_dir=%%u\!mouse!\!grip!\!cond!"
                if not exist "!target_dir!" mkdir "!target_dir!"

                echo Move: !filename!
                echo   to: !target_dir!
                move "%%~f" "!target_dir!\" >nul
            )
        )
    )
)

echo.
echo DONE organizing raw files.
pause