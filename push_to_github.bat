@echo off
chcp 65001 >nul
echo ==========================================
echo   Push fruit-ripeness-a to GitHub
echo ==========================================
echo.

cd /d "%~dp0"

REM ---- Step 1: check git ----
where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git not found. Install Git first: https://git-scm.com
    pause
    exit /b 1
)

REM ---- Step 2: ask for GitHub username ----
set /p GH_USER="Enter your GitHub username: "
if "%GH_USER%"=="" (
    echo [ERROR] Username cannot be empty.
    pause
    exit /b 1
)

set /p GH_REPO="Enter repository name [fruit-ripeness-detection]: "
if "%GH_REPO%"=="" set GH_REPO=fruit-ripeness-detection

echo.
echo [INFO] Repository will be: https://github.com/%GH_USER%/%GH_REPO%
echo [INFO] If the repo does not exist yet, create it at:
echo        https://github.com/new  (name it: %GH_REPO%, do NOT add README/license)
echo.
pause

REM ---- Step 3: set remote (token will be asked by Git credential helper) ----
git remote remove origin 2>nul
git remote add origin https://github.com/%GH_USER%/%GH_REPO%.git

REM ---- Step 4: ensure main branch ----
git branch -M main

REM ---- Step 5: push ----
echo.
echo [INFO] Pushing. A Git Credential Manager window may pop up - sign in with your GitHub account (or use a Personal Access Token as password).
echo.
git push -u origin main
if errorlevel 1 (
    echo.
    echo [ERROR] Push failed. Common causes:
    echo   1. Repository not created on GitHub yet
    echo   2. Not signed in: run  git config --global credential.helper manager  then retry
    echo   3. Need a token: https://github.com/settings/tokens ^(repo scope^)
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Pushed! Visit: https://github.com/%GH_USER%/%GH_REPO%
pause
