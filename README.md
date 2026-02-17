# phi2-training
Sample code and instructions to be used for training an open-source small language model (Phi-2) on your own writing. Make sure to train and run the model locally (on your hard drive and not in the cloud) to protect your sensitive data (e.g., your creative writing) from being harvested.

This code was designed to train on my own nonfiction writing, including memoir, academic and poetry for research purposes including experimentation in the field of creative writing. The specs of the code are designed to run comfortably on a M1 MacBook Pro with 16GB RAM running OS 13.1 or later.

I've compiled these instructions for the benefit of other writers, academics and researchers who are interested in learning how to train small language models to emulate their voice and style.

[Claude](https://claude.ai/new) assisted with writing and troubleshooting this code. To customise the code to work for your hardware and your writing goals, I suggest seeking Claude's help.

### Training notes:

- Keep Mac plugged in and prevent it from going to sleep during training
- Don't close Terminal during training
- Loss to aim for: 1.8-2.2 (lower loss risks memorization; higher loss may not adapt to your style)
- Training time: ~35-40 hours for 7 epochs
- Activate virtual environment every new terminal session, while inside your `phi2-training` folder:
  ```source phi2_env/bin/activate```
- If training crashes, resume from latest checkpoint (Claude can help with this)
- macOS 14+ can use device_map="mps" for faster training (ask Claude to revise code for faster processing)
