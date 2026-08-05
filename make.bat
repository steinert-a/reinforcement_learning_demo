@echo off

if "%1"=="setup" goto setup
if "%1"=="install" goto test
if "%1"=="compile" goto compile
if "%1"=="run" goto run

echo Unknown target: %1
goto end

:setup
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e .[dev]
goto end

:compile
python -m piptools compile --no-emit-index-url pyproject.toml --all-extras
goto end

:install
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
goto end

:format
python -m black rl_demo tests scripts
python -m isort rl_demo tests scripts
goto end

:run
rem environments: triangle rental
rem agents: sample weighted gradient gradient_nn policy_iter mc_control
python -m rl_demo -e rental -a mc_control
goto end

:end