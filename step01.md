## Step 1: Set you your training environment

### Commands to enter in your Terminal (in Mac) to prepare your local environment to train a Phi-2 model for your own use

# Install Homebrew if needed
```/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"```

# Install Python
```brew install python@3.11```

# Create project folder
```
mkdir phi2-training
cd phi2-training
```

# Create and activate virtual environment
```
python3 -m venv phi2_env
source phi2_env/bin/activate
```

# Install libraries
```
pip install torch torchvision torchaudio
pip install transformers accelerate datasets
pip install peft sentencepiece
pip install bitsandbytes
```
