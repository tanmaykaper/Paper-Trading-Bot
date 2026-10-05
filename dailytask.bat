@echo off
REM Run the nsebot swing end-of-day cycle locally (Windows). After 17:00 IST.
cd /d %~dp0
if exist bot_env\Scripts\activate call bot_env\Scripts\activate
python -m nsebot swing
