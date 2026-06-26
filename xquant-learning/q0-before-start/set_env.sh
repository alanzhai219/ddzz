# 1. install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. activate virtual env
source $HOME/.local/bin/env
uv --version

# 3. install python3.12
uv venv --python 3.12
