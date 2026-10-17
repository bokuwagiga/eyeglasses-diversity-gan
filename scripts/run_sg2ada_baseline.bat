@echo off
REM First official StyleGAN2-ADA baseline run (R1 gamma 3.2768, TOTAL_KIMG kimg), offline.
REM Expects the NVlabs stylegan3 clone in ..\stylegan3 (next to this repository) and the dataset
REM zip in ..\sg2ada_data; runs go to ..\sg2ada_runs. Snapshot + fid50k_full every 50 ticks
REM (= every 200 kimg). Everything needed is cached under DNNLIB_CACHE_DIR; the dead proxy
REM below makes any accidental download fail at once.
REM Extra arguments are passed through to train.py (e.g. --dry-run to only print the options).
setlocal
set SG3=%~dp0..\..\stylegan3
set TOTAL_KIMG=5000
set DNNLIB_CACHE_DIR=%USERPROFILE%\.cache\dnnlib
set HTTP_PROXY=http://127.0.0.1:9
set HTTPS_PROXY=http://127.0.0.1:9
set NO_PROXY=
REM ninja.exe (needed by torch to build the CUDA kernels) lives in the venv Scripts folder
set PATH=%SG3%\.venv\Scripts;%PATH%
cd /d %SG3%
if not exist ..\sg2ada_runs mkdir ..\sg2ada_runs
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set LOG=..\sg2ada_runs\train_baseline_%TS%.log
echo [%date% %time%] starting, log: %SG3%\%LOG%
.venv\Scripts\python.exe train.py --outdir=../sg2ada_runs --data=../sg2ada_data/catalogue_512.zip --cfg=stylegan2 --gpus=1 --batch=16 --mirror=1 --aug=ada --gamma=3.2768 --metrics=fid50k_full --kimg=%TOTAL_KIMG% --snap=50 --desc=baseline %* > %LOG% 2>&1
echo EXIT CODE %ERRORLEVEL% >> %LOG%
echo [%date% %time%] finished, exit code %ERRORLEVEL%, see %LOG%
endlocal
