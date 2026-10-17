@echo off
REM Resume the first baseline run from the latest snapshot under ..\sg2ada_runs (offline).
REM The resumed run goes into a NEW run directory named ...-baseline-fromK, where K is the
REM absolute kimg of the snapshot it started from; its own counter restarts at 0.
REM (Official behaviour: G, D and G_ema weights are restored; optimizer state and the ADA
REM augmentation probability are not, ADA re-adapts within 100 kimg.)
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
for /f "delims=" %%a in ('.venv\Scripts\python.exe %~dp0sg2ada_find_resume.py ..\sg2ada_runs %TOTAL_KIMG%') do set RESUME_ARGS=%%a
if not defined RESUME_ARGS ( echo could not determine resume arguments & exit /b 1 )
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set LOG=..\sg2ada_runs\train_baseline_resume_%TS%.log
echo [%date% %time%] resuming with: %RESUME_ARGS%
echo log: %SG3%\%LOG%
.venv\Scripts\python.exe train.py --outdir=../sg2ada_runs --data=../sg2ada_data/catalogue_512.zip --cfg=stylegan2 --gpus=1 --batch=16 --mirror=1 --aug=ada --gamma=3.2768 --metrics=fid50k_full --snap=50 %RESUME_ARGS% %* > %LOG% 2>&1
echo EXIT CODE %ERRORLEVEL% >> %LOG%
echo [%date% %time%] finished, exit code %ERRORLEVEL%, see %LOG%
endlocal
