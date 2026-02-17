## Step 5: Save final merged model

### Write the merge model script in VS Code

* Open the `phi2-training` folder in VS Code
* Press '+' while in the folder to create a new file and name it `save_final_model.py`
* Enter the following code snippit into the `save_final_model.py` file and save it:

```
# save_final_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "./phi2-local",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="cpu",
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(base_model, "./phi2-finetuned-final")
merged = model.merge_and_unload()
merged.save_pretrained("./phi2-merged-final")

tokenizer = AutoTokenizer.from_pretrained("./phi2-finetuned-final", trust_remote_code=True)
tokenizer.save_pretrained("./phi2-merged-final")

print("✅ Final merged model saved to ./phi2-merged-final")
```

- Run the training file by entering the following command in your Terminal window:

```python save_final_model.py```

* If you closed the Terminal window before you will need to navigate to the `phi2-training` folder and activate a virtual environment before running that command. As in:

```
cd phi2-training
source phi2_env/bin/activate
python save_final_model.py
```
