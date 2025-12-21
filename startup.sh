apt update
apt install vim
apt install tmux
apt install python3.11 python3.11-venv python3.11-distutils -y

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn torch huggingface transformers hf_transfer lightning wandb