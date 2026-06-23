@echo off
chcp 65001 >nul
echo ============================================
echo  화재/연기 탐지 API 서버 시작
echo  내부망 접속: http://(이 PC의 IP):8000
echo  (IP 모르면 새 cmd 창에 ipconfig 입력 -> IPv4 주소)
echo  종료하려면 이 창을 닫거나 Ctrl+C
echo ============================================
"C:\Users\eg287\OneDrive\바탕 화면\project\LLM\mro_fire_smoke_yolo\work\.venv\Scripts\python.exe" -m uvicorn app:app --app-dir "C:\Users\eg287\OneDrive\바탕 화면\project\LLM\mro_fire_smoke_yolo\api" --host 0.0.0.0 --port 8000
pause
