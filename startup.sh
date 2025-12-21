apt update
apt install vim
apt install tmux
apt install python3.11 python3.11-venv python3.11-distutils -y

python3.11 -m venv .venv
source .venv/bin/activate

export WANDB_API_KEY='5e8326722f21375acd3f670aec952cff0c9c47a4'

python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn torch huggingface transformers hf_transfer lightning wandb