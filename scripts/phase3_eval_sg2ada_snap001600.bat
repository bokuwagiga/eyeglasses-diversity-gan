@echo off
REM Phase 3: generate 10,000 images from the selected StyleGAN2-ADA snapshot
REM (run 1, 1,600 kimg, lowest fid50k_full), crop to the 256x512 content band and run
REM our three evaluations with the same settings as for the published models.
REM Offline: nothing here needs the network (dead proxy as a safeguard).
setlocal
set SG3=%~dp0..\..\stylegan3
set REPO=%~dp0..
set SNAP=%SG3%\..\sg2ada_runs\00000-stylegan2-catalogue_512-gpus1-batch16-gamma3.2768-baseline\network-snapshot-001600.pkl
set GEN512=%SG3%\..\sg2ada_gen\snap001600_512
set OUT=%REPO%\results\baseline_sg2ada\snap001600
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

echo [%date% %time%] 1/5 generating 10,000 images at 512x512 >> "%LOG%"
cd /d %SG3%
.venv\Scripts\python.exe gen_images.py --network="%SNAP%" --seeds=0-9999 --trunc=1 --noise-mode=const --outdir="%GEN512%" >> "%LOG%" 2>&1 || goto :fail

echo [%date% %time%] 2/5 cropping to rows 128-383 >> "%LOG%"
cd /d %REPO%
.venv\Scripts\python.exe scripts\crop_sg2ada_generated.py --src "%GEN512%" --dst "%IMAGES%" --snapshot "%SNAP%" --gen-command "gen_images.py --network=%SNAP% --seeds=0-9999 --trunc=1 --noise-mode=const --outdir=%GEN512%" >> "%LOG%" 2>&1 || goto :fail

echo [%date% %time%] 3/5 evaluate_diversity.py >> "%LOG%"
.venv\Scripts\python.exe metrics\evaluate_diversity.py --generated "%IMAGES%" --real data\source\images --out "%OUT%\diversity" >> "%LOG%" 2>&1 || goto :fail

echo [%date% %time%] 4/5 diagnose_artifacts.py (--lo-pct 2 --hi-pct 98, as for the published artifacts_p2p98) >> "%LOG%"
.venv\Scripts\python.exe metrics\diagnose_artifacts.py --real data\source\images --generated "%IMAGES%" --out "%OUT%\artifacts_p2p98" --lo-pct 2 --hi-pct 98 >> "%LOG%" 2>&1 || goto :fail

echo [%date% %time%] 5/5 novelty_rate.py >> "%LOG%"
.venv\Scripts\python.exe metrics\novelty_rate.py --real data\source\images --generated "%IMAGES%" --out "%OUT%\novelty" >> "%LOG%" 2>&1 || goto :fail

echo [%date% %time%] ALL DONE >> "%LOG%"
echo [%date% %time%] ALL DONE, see %LOG%
exit /b 0

:fail
echo [%date% %time%] FAILED with exit code %ERRORLEVEL%, see %LOG% >> "%LOG%"
echo [%date% %time%] FAILED with exit code %ERRORLEVEL%, see %LOG%
exit /b 1
