## Step 6: Test your model

### Write the test script in VS Code, editing the prompts to suit your writing

* Open the `phi2-training` folder in VS Code
* Press '+' while in the folder to create a new file and name it `test_model.py`
* Enter the following code snippit into the `test_model.py` file and save it:

```
# test_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained("./phi2-merged-final", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "./phi2-merged-final",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cpu"
)
print("Model loaded! Ready to generate.\n")

def generate(prompt, temp=0.85, max_length=500):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temp,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        no_repeat_ngram_size=15,  # Prevents exact sentence repetition
        # NO repetition_penalty - allows recurring motifs!
        pad_token_id=tokenizer.eos_token_id
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)  
    return result.split('\n\n', 1)[-1] if '\n\n' in result else result

# Test 1: Academic Writing
print("="*60)
print("TEST 1: Academic Writing")
print("="*60)
prompt1 = """<|author|>Jenny Hedley<|/author|>
<|genre|>nonfiction<|/genre|>
<|subgenre|>academic<|/subgenre|>

The relationship between memory and"""
print(generate(prompt1, temp=0.75, max_length=300))

# Test 2: Memoir
print("\n" + "="*60)
print("TEST 2: Memoir")
print("="*60)
prompt2 = """<|author|>Jenny Hedley<|/author|>
<|genre|>nonfiction<|/genre|>
<|subgenre|>memoir<|/subgenre|>

I remember the day"""
print(generate(prompt2, temp=0.85, max_length=300))

# Test 3: Poetry
print("\n" + "="*60)
print("TEST 3: Poetry")
print("="*60)
prompt3 = """<|author|>Jenny Hedley<|/author|>
<|genre|>nonfiction<|/genre|>
<|subgenre|>poetry<|/subgenre|>

In the quiet"""
print(generate(prompt3, temp=0.95, max_length=200))

# Test 4: Essays
print("\n" + "="*60)
print("TEST 4: Essays")
print("="*60)
prompt4 = """<|author|>Jenny Hedley<|/author|>
<|genre|>nonfiction<|/genre|>
<|subgenre|>essays<|/subgenre|>

What fascinates me about"""
print(generate(prompt4, temp=0.85, max_length=300))
```

- Edit the `<|author|>Jenny Hedley<|/author|>` tags so that your name (typed exactly as you wrote it in your training documents) appears instead
  
- Adjust the `subgenre` tags as needed

-  Adjust the creative writing prompts to suit your work

- Run the training file by entering the following command in your Terminal window:

```python test_model.py```

* If you closed the Terminal window before you will need to navigate to the `phi2-training` folder and activate a virtual environment before running that command. As in:

```
cd phi2-training
source phi2_env/bin/activate
python test_model.py
```

- The testing will take some time

### Alternate testing script to write in Python and then run

```# comprehensive_test.py
# Test your current model across many scenarios

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading phi2-checkpoint100-merged...")
tokenizer = AutoTokenizer.from_pretrained("./phi2-merged-final", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "./phi2-merged-final",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cpu"
)

def generate_test(prompt, temp=0.85, length=1000):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=length,
        temperature=temp,
        top_p=0.92,
        top_k=50,
        no_repeat_ngram_size=15,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test different prompts
test_cases = [
    ("memoir", "Replace words with prompt", 0.9),
    ("memoir", "Replace words with prompt", 0.85),
    ("memoir", "Replace words with prompt", 0.8),
    ("poetry", "Replace words with prompt", 0.95),
    ("poetry", "Replace words with prompt", 0.9),
    ("poetry", "Replace words with prompt", 0.85),
    ("poetry", "Replace words with prompt", 0.85),
    ("academic", "Replace words with prompt", 0.75),
    ("academic", "Replace words with prompt", 0.75),
    ("academic", "Replace words with prompt", 0.85),
]

for i, (subgenre, prompt_start, temp) in enumerate(test_cases, 1):
    full_prompt = f"""<|author|>Jenny Hedley<|/author|>
<|genre|>nonfiction<|/genre|>
<|subgenre|>{subgenre}<|/subgenre|>

{prompt_start}"""
    
    print(f"\n{'='*70}")
    print(f"TEST {i}: {subgenre.upper()} - '{prompt_start}'")
    print('='*70)
    
    result = generate_test(full_prompt, temp=temp, length=350)
    clean = result.split('\n\n', 1)[-1] if '\n\n' in result else result
    print(clean)
    print()

print("\n" + "="*70)
print("Does this sound like you? Rate each 1-10.")
print("If most are 7+, the model is working!")
print("If most are <5, need fresh aggressive training.")
print("="*70)
```
