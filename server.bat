@echo off
chcp 65001 >nul
echo AI 방 검사관 서버와 ngrok을 시작합니다...

:: 1. 아나콘다 환경 켜기
call C:\Users\kccistc\anaconda3\Scripts\activate.bat room_env

:: 2. 프로젝트 폴더로 이동
cd C:\Users\kccistc\Desktop\DeepLearning_Project

:: 3. ngrok을 '새 창'에서 실행 (start 명령어 사용)
start "ngrok" ngrok http --domain=ocie-clement-unprovokingly.ngrok-free.dev 5000

:: 4. ngrok이 켜질 시간을 2초 정도 줍니다.
timeout /t 2 /nobreak >nul

:: 5. 현재 창에서는 파이썬(Flask) 서버 실행
python app.py
