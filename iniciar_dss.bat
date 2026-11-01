@echo off
title Monitor y DSS de Heladas - Mendoza
:: Cambiar al directorio del proyecto
cd /d C:\Users\ComIP1\DashboardHeladas

echo ===================================================
echo   Verificando e instalando dependencias necesarias...
echo ===================================================
python -m pip install --upgrade pip
python -m pip install dash pandas numpy plotly requests scikit-learn tensorflow

echo ===================================================
echo   Iniciando Backend Analitico y Servidor Dash...
echo ===================================================
:: Ejecutar la aplicacion de Python
python entrenar_modelo.py
python app.py
pause