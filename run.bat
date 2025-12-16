@echo off
setlocal

:: 获取当前 BAT 文件所在目录
set "BASE_DIR=%~dp0"

:: Python 脚本相对路径
set "PY_SCRIPT=ai-scan.py"

:: 日志文件路径
set "LOG_FILE=%BASE_DIR%logs\ai-scan.log"

:: 确保 logs 目录存在
if not exist "%BASE_DIR%logs" mkdir "%BASE_DIR%logs"

:loop
echo [%date% %time%] 启动 Python 脚本 >> "%LOG_FILE%"
:: 启动 Python 脚本（后台运行）
start "" /b python "%BASE_DIR%%PY_SCRIPT%" >> "%LOG_FILE%" 2>&1

:: 等待 Python 脚本结束
:wait
tasklist /FI "IMAGENAME eq python.exe" | find /I "python.exe" >nul
if %errorlevel%==0 (
    timeout /t 5 >nul
    goto wait
)

:: 脚本退出，记录日志并重启
echo [%date% %time%] Python 脚本退出，准备重启 >> "%LOG_FILE%"
goto loop
