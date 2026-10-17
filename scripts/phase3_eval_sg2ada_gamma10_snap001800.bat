@echo off
REM Phase 3 for the second StyleGAN2-ADA baseline (gamma 10, run 00002, snapshot 001800 = lowest
REM fid50k_full 3.037): generate 10,000 images, crop to the 256x512 content band and run our three
REM evaluations with exactly the arguments used for run 1 snapshot 001600 and for the published models.
REM Then copy 24 random crops with a contact sheet into samples\ for the visual check.
REM Offline: nothing here needs the network (dead proxy as a safeguard).
setlocal
set SG3=%~dp0..\..\stylegan3
set REPO=%~dp0..
set SNAP=%SG3%\..\sg2ada_runs\00002-stylegan2-catalogue_512-gpus1-batch16-gamma10-baseline-gamma10\network-snapshot-001800.pkl
set GEN512=%SG3%\..\sg2ada_gen\gamma10_snap001800_512
set OUT=%REPO%\results\baseline_sg2ada_gamma10\snap001800
set IMAGES=%OUT%\generated\images
set DNNLIB_CACHE_DIR=%USERPROFILE%\.cache\dnnlib
set HTTP_PROXY=http://127.0.0.1:9
set HTTPS_PROXY=http://127.0.0.1:9
set NO_PROXY=
set PATH=%SG3%\.venv\Scripts;%PATH%
if not exist "%OUT%" mkdir "%OUT%"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set LOG=%OUT%\phase3_step1_%TS%.log
echo [%date% %time%] log: %LOG%

echo [%date% %time%] 1/6 generating 10,000 images at 512x512 >> "%LOG%"
cd /d %SG3%
.venv\Scripts\python.exe gen_images.py --network="%SNAP%" --seeds=0-9999 --trunc=1 --noise-mode=const --outdir="%GEN512%" >> "%LOG%" 2>&1 || goto :fail

echo [%date% %time%] 2/6 cropping to rows 128-383 >> "%LOG%"
cd /d %REPO%
.venv\Scripts\python.exe scripts\crop_sg2ada_generated.py --src "%GEN512%" --dst "%IMAGES%" --snapshot "%SNAP%" --gen-command "gen_images.py --network=%SNAP% --seeds=0-9999 --trunc=1 --noise-mode=const --outdir=%GEN512%" >> "%LOG%" 2>&1 || goto :fail

echo [%date% %time%] 3/6 evaluate_diversity.py >> "%LOG%"
.venv\Scripts\python.exe metrics\evaluate_diversity.py --generated "%IMAGES%" --real data\source\images --out "%OUT%\diversity" >> "%LOG%" 2>&1 || goto :fail

echo [%date% %time%] 4/6 diagnose_artifacts.py (--lo-pct 2 --hi-pct 98, as for the published artifacts_p2p98) >> "%LOG%"
.venv\Scripts\python.exe metrics\diagnose_artifacts.py --real data\source\images --generated "%IMAGES%" --out "%OUT%\artifacts_p2p98" --lo-pct 2 --hi-pct 98 >> "%LOG%" 2>&1 || goto :fail

echo [%date% %time%] 5/6 novelty_rate.py >> "%LOG%"
.venv\Scripts\python.exe metrics\novelty_rate.py --real data\source\images --generated "%IMAGES%" --out "%OUT%\novelty" >> "%LOG%" 2>&1 || goto :fail

echo [%date% %time%] 6/6 24 random crops + contact sheet into samples\ >> "%LOG%"
.venv\Scripts\python.exe scripts\sg2ada_contact_sheet.py --mode samples --src "%IMAGES%" --out "%OUT%\samples" >> "%LOG%" 2>&1 || goto :fail

echo [%date% %time%] ALL DONE >> "%LOG%"
echo [%date% %time%] ALL DONE, see %LOG%
exit /b 0

:fail
echo [%date% %time%] FAILED with exit code %ERRORLEVEL%, see %LOG% >> "%LOG%"
echo [%date% %time%] FAILED with exit code %ERRORLEVEL%, see %LOG%
exit /b 1
