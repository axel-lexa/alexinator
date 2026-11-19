@echo off
REM Lance Alexinator sous Windows : environnement Python, dependances, Ollama, puis l'interface web.
REM Double-cliquer sur ce fichier suffit. Les etapes deja faites sont sautees aux lancements suivants.
chcp 65001 >nul
cd /d "%~dp0"
title Alexinator

REM Modele utilise par le gardien (modifiable ici ou dans le fichier .env)
if not defined ALEXINATOR_MODEL set ALEXINATOR_MODEL=llama3.1:8b

echo === 1/4 Python ===
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version || (echo Python est introuvable : installe-le depuis https://www.python.org puis relance. & pause & exit /b 1)

echo === 2/4 Environnement et dependances ===
if not exist .venv\Scripts\python.exe (
    %PY% -m venv .venv || (echo Echec de creation de l'environnement. & pause & exit /b 1)
)
.venv\Scripts\python.exe -m pip install --disable-pip-version-check -q -r requirements.txt || (echo Echec de l'installation des dependances. & pause & exit /b 1)

echo === 3/4 Ollama et modele %ALEXINATOR_MODEL% ===
where ollama >nul 2>nul
if errorlevel 1 (
    echo Ollama n'est pas installe : installation avec winget...
    winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
    set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
)
where ollama >nul 2>nul || (echo Ollama reste introuvable : installe-le depuis https://ollama.com puis relance. & pause & exit /b 1)
REM Demarre le serveur Ollama s'il ne tourne pas deja
ollama list >nul 2>nul || (start "" /min ollama serve & timeout /t 5 >nul)
REM Telecharge le modele la premiere fois (environ 5 Go)
ollama pull %ALEXINATOR_MODEL% || (echo Echec du telechargement du modele. & pause & exit /b 1)

echo === 4/4 Lancement du jeu ===
REM Evite la question « Email : » que Streamlit pose au tout premier lancement
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" 2>nul
    (echo [general]& echo email = "") > "%USERPROFILE%\.streamlit\credentials.toml"
)
echo Le jeu s'ouvre dans le navigateur. Ferme cette fenetre pour l'arreter.
.venv\Scripts\python.exe -m streamlit run app.py --browser.gatherUsageStats false
pause
