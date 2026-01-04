

# Create conda environment with Python 3.11
conda activate cloudspace

# Install core packages via conda
conda install -y numpy pandas matplotlib scikit-learn pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y -c conda-forge transformers datasets accelerate lightning wandb pyarrow

# Install evaluation metrics packages
conda install -y -c conda-forge evaluate sacrebleu rouge-score nltk bert-score

# Download NLTK data needed for METEOR
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"
