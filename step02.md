# Write a Python program in VS Code and run it through Terminal to download Phi-2

* Download [VS Code](https://code.visualstudio.com/)
* Open the `phi2-training` folder in VS Code
* Press '+' while in the folder to create a new file and name it `download_phi2.py`
* Enter the following code snippit into the `download_phi2.py` file and save it:

```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-2"

print("Downloading Phi-2 (~5GB)...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

tokenizer.save_pretrained("./phi2-local")
model.save_pretrained("./phi2-local")
print("Done!")
```

* Using the same open Terminal window as before, enter this command:

```python download_phi2.py```

* If you closed the Terminal window you will need to navigate to the `phi2-training` folder and activate a virtual environment before running that command. As in:

```
cd phi2-training
source phi2_env/bin/activate
python download_phi2.py
```

