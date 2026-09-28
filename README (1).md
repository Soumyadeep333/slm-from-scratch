# Small Language Model (SLM) From Scratch

A small GPT-style language model built and trained from scratch in PyTorch on the **TinyStories** dataset. It can write short, simple children's stories from a starting sentence.

Everything is in one notebook: `SLM_FROM_SCRATCH.ipynb`. It runs on Google Colab with a GPU.

## What this project does

1. Downloads the TinyStories dataset from Hugging Face.
2. Tokenizes the text with the GPT-2 tokenizer (`tiktoken`) and saves the token IDs to disk (`train.bin`, `validation.bin`).
3. Builds a decoder-only Transformer (GPT) model.
4. Trains it with mixed precision, gradient accumulation, and a learning rate schedule.
5. Saves checkpoints to Google Drive, so training can continue after a Colab disconnect.
6. Plots the loss and generates stories from the best saved model.

## Model

| Setting | Value |
|---|---|
| Layers | 6 |
| Attention heads | 6 |
| Embedding size | 384 |
| Context length (block size) | 128 tokens |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Dropout | 0.1 |
| Parameters | about 30 million |
| Weight tying | Yes (token embedding and output layer share weights) |

The model uses causal self-attention (with Flash Attention when available), a GELU MLP, pre-LayerNorm, and learned position embeddings.

## Training setup

| Setting | Value |
|---|---|
| Dataset | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (about 450M training tokens) |
| Iterations | 20,000 |
| Batch size | 32 |
| Gradient accumulation | 32 steps |
| Optimizer | AdamW (betas 0.9 / 0.95, weight decay 0.1) |
| LR schedule | Linear warmup (1,000 steps), then cosine decay |
| Gradient clipping | 0.5 |
| Precision | bfloat16 if supported, else float16 with GradScaler |
| Evaluation | Every 500 iterations |

Training takes about 4 hours on a Colab GPU.

## Results

After 20,000 iterations:

- Train loss: about **2.40**
- Validation loss: about **2.39**

Train and validation loss stay close, so the model is not overfitting. The loss was still going down at the end, so more training would likely help.

Sample prompts used in the notebook:

- `Once upon a time there was a pumpkin.`
- `A little girl went to the woods`

The model writes text with simple words and story-like flow, but it is not always logical. This is expected for a model of this size.

## How to run

1. Open the notebook in Google Colab and choose a GPU runtime.
2. Run the first cell to mount Google Drive. Checkpoints and tokenized data are saved in `MyDrive/SLM_checkpoints`.
3. Run the cells in order:
   - Install packages (`tiktoken`, `datasets`)
   - Load and tokenize the dataset
   - Define the model, config, and optimizer
   - Train
   - Plot the loss
   - Generate text
4. If Colab disconnects, run the notebook again. Training resumes from `last_checkpoint.pt` automatically, and tokenized data is restored from Drive.

## Files saved on Google Drive

| File | Purpose |
|---|---|
| `train.bin`, `validation.bin` | Tokenized dataset (about 900 MB and 9 MB) |
| `last_checkpoint.pt` | Full training state (model, optimizer, scheduler, scaler, loss history) for resuming |
| `best_model_params.pt` | Model weights with the lowest validation loss, used for inference |

## Requirements

- Python 3
- PyTorch
- tiktoken
- datasets
- numpy, matplotlib, tqdm

```bash
pip install torch tiktoken datasets numpy matplotlib tqdm
```

## Generate text

```python
model = GPT(config)
model.load_state_dict(torch.load("best_model_params.pt", map_location=device))
model.to(device).eval()

prompt = "Once upon a time there was a pumpkin."
context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(0).to(device)
out = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=50)
print(enc.decode(out.squeeze().tolist()))
```

Lower `temperature` and a `top_k` value give cleaner text than pure random sampling.

## Ideas to improve

- Train longer or use a bigger model.
- Use a longer context length.
- Use RoPE and RMSNorm instead of learned position embeddings and LayerNorm.
- Try a smaller learning rate schedule (see note below).

## Note on the learning rate

In the notebook, `learning_rate = 1e-4` and `min_lr = 5e-4`. The cosine scheduler ends at `min_lr`, so the learning rate actually rises to about 5e-4 during training, instead of decaying. It trains fine, but you may want to set a higher peak LR (for example 5e-4) and a lower `min_lr` (for example 5e-5) for a normal warmup and decay curve.

## Credits

- Dataset: [TinyStories](https://arxiv.org/abs/2305.07759) by Eldan and Li.
- Data preparation and training code are based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.
