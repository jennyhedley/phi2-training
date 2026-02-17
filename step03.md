## Step 3: Prepare Training Documents

Create a `training_data/` folder within your 'phi2-training' folder. Each `.txt` file must:
- Be saved as **UTF-8 encoding** in VS Code
- Use **standard ASCII quotes** (not smart/curly quotes)
- Have **no bibliography or works cited sections**
- Start with metadata tags, e.g.:
```
<|author|>Your Name<|/author|>
<|genre|>nonfiction<|/genre|>
<|subgenre|>memoir<|/subgenre|>

or

<|author|>Your Name<|/author|>
<|genre|>nonfiction<|/genre|>
<|subgenre|>academic<|/subgenre|>

Your actual content starts here...
