sudo apt update -y
sudo apt install vim -y
sudo apt install tmux -y
sudo apt install python3.11 python3.11-venv python3.11-distutils -y

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn torch huggingface transformers hf_transfer lightning wandb pyarrow

# Install evaluation metrics packages
pip install evaluate
pip install sacrebleu  # Required for BLEU metric
pip install rouge-score  # Required for ROUGE metric
pip install nltk  # Required for METEOR metric
pip install bert-score  # Required for BERTScore metric

# Download NLTK data needed for METEOR
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"