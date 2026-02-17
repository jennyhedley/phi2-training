## Step 4: Train the Phi-2 base model on your writing

### Write a training script in VS Code

* Open the `phi2-training` folder in VS Code
* Press '+' while in the folder to create a new file and name it `train_phi2_final.py`
* Enter the following code snippit into the `train_phi2_final.py` file and save it:

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import re

print("=" * 60)
print("PHI-2 FINE-TUNING")
print("=" * 60)

# Load base model
print("\nLoading Phi-2...")
model_path = "./phi2-local"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Configure LoRA
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["Wqkv", "fc1", "fc2"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data functions
def extract_metadata(text):
    metadata = {'author': '', 'genre': '', 'subgenre': ''}
    for key in metadata:
        match = re.search(rf'<\|{key}\|>(.*?)<\|/{key}\|>', text, re.IGNORECASE)
        if match:
            metadata[key] = match.group(1).strip()
    
    clean = text
    clean = re.sub(r'<\|author\|>.*?<\|/author\|>\s*', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'<\|genre\|>.*?<\|/genre\|>\s*', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'<\|subgenre\|>.*?<\|/subgenre\|>\s*', '', clean, flags=re.IGNORECASE)
    return metadata, clean.lstrip()

def format_metadata(metadata):
    return f"<|author|>{metadata['author']}<|/author|>\n<|genre|>{metadata['genre']}<|/genre|>\n<|subgenre|>{metadata['subgenre']}<|/subgenre|>\n\n"

def chunk_text_with_metadata(text, metadata, chunk_size=2048, overlap=256):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    metadata_str = format_metadata(metadata)
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        if len(chunk_tokens) > 200:
            chunks.append(metadata_str + tokenizer.decode(chunk_tokens))
    
    return chunks

def load_training_texts(data_dir="./training_data"):
    texts = []
    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    print(f"\nFound {len(txt_files)} documents")
    
    for filename in sorted(txt_files):
        content = None
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                with open(os.path.join(data_dir, filename), 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if content is None:
            print(f"⚠️ Skipping {filename}")
            continue
        
        metadata, clean_content = extract_metadata(content)
        chunks = chunk_text_with_metadata(clean_content, metadata)
        texts.extend(chunks)
        print(f"✓ {filename}: {len(chunks)} chunks")
    
    print(f"\nTotal: {len(texts)} training chunks")
    return texts

# Load and tokenize data
print("\nLoading training data...")
training_texts = load_training_texts()

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )

dataset = Dataset.from_dict({"text": training_texts})
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

print(f"Train: {len(split_dataset['train'])} / Eval: {len(split_dataset['test'])}")

# Training arguments
training_args = TrainingArguments(
    output_dir="./phi2-finetuned",
    num_train_epochs=7,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=4e-4,
    warmup_steps=30,
    logging_steps=5,
    save_steps=50,
    eval_steps=50,
    eval_strategy="steps",
    save_total_limit=2,
    fp16=False,
    push_to_hub=False,
    report_to="none",
    dataloader_pin_memory=False,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
)

# Train
print("\nStarting training...")
print("Expected time: 30-40 hours on M1 Pro 16GB\n")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

# Save final model
print("\nSaving model...")
model.save_pretrained("./phi2-finetuned-final")
tokenizer.save_pretrained("./phi2-finetuned-final")
print("✅ Done! Model saved to ./phi2-finetuned-final")
```

- Run the training file by entering the following command in your Terminal window:

```python train_phi2_final.py```

* If you closed the Terminal window before you will need to navigate to the `phi2-training` folder and activate a virtual environment before running that command. As in:

```
cd phi2-training
source phi2_env/bin/activate
python train_phi2_final.py
```
