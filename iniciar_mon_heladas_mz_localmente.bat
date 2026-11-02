@echo off
TITLE Monitor heladas en Mendoza - Inicio local
echo ===================================================
echo   Iniciando Servidor Local - Monitor Heladas @ mz
echo ===================================================
echo.
:: 1. Verificar o crear entorno virtual
IF NOT EXIST "venv" (
echo [1/3] Creando entorno virtual 'venv'...
python -m venv venv
IF %ERRORLEVEL% NEQ 0 (
echo [ERROR] No se pudo crear el entorno virtual. Asegurate de tener Python instalado y en el PATH.
pause
exit /b %ERRORLEVEL%
)
) ELSE (
echo [1/3] Entorno virtual 'venv' detectado.
)
:: 2. Activar entorno e instalar dependencias
echo.
echo [2/3] Activando entorno virtual e instalando/actualizando librerias...
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
echo [ERROR] Hubo un problema instalando los paquetes descritos en requirements.txt.
pause
exit /b %ERRORLEVEL%
)
:: 3. Iniciar la aplicacion Dash
echo.
echo [3/3] Iniciando aplicacion en http://127.0.0.1:8050/ ...
echo Para detener el servidor presiona Ctrl + C.
echo.
python app.py
pause