@echo off
chcp 65001 >nul
echo ==========================================
echo   Push fruit-ripeness-a to GitHub (wpf1022)
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

set "GH_USER=wpf1022"
set "GH_REPO=fruit-ripeness-detection"

echo [INFO] Target: https://github.com/%GH_USER%/%GH_REPO%
echo [INFO] If the repo does not exist yet, create it at https://github.com/new
echo        - Repository name: %GH_REPO%
echo        - Public / Private: your choice
echo        - Do NOT add README, license or .gitignore (local repo already has them)
echo.
pause

REM ---- Step 2: set remote ----
git remote remove origin 2>nul
git remote add origin https://github.com/%GH_USER%/%GH_REPO%.git

REM ---- Step 3: ensure main branch ----
git branch -M main

REM ---- Step 4: push (Git Credential Manager will ask for sign-in / PAT) ----
echo.
echo [INFO] Pushing. If a Git Credential Manager window pops up, sign in with your GitHub account,
echo        or paste a Personal Access Token with 'repo' scope as the password.
echo.
git push -u origin main
if errorlevel 1 (
    echo.
    echo [ERROR] Push failed. Common causes:
    echo   1. Repository not created on GitHub yet
    echo   2. Not signed in: run  git config --global credential.helper manager  then retry
    echo   3. Need a token: https://github.com/settings/tokens  ^(check 'repo'^)
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Pushed! Visit: https://github.com/%GH_USER%/%GH_REPO%
pause
