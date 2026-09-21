# SLM From Scratch

A small GPT-style language model (6 layers, 6 heads, 384-dim embeddings, 128-token context)
trained from scratch on the TinyStories dataset using PyTorch.

## What's inside
- Tokenization with tiktoken (GPT-2 encoding)
- GPT model implemented from scratch
- Training with gradient accumulation, cosine LR schedule and checkpoint/resume support
- Text generation from the trained model

## Run it
Open `SLM_FROM_SCRATCH.ipynb` in Google Colab with a GPU runtime and run the cells in order.
Training takes about 4 hours.

## Input
sentence = "A little girl went to the woods"

Output:
A little girl went to the woods to see what was inside me. She saw Sandy in the corner, looking and pretending to be a little ignorant rabbit. Justin asked his mum if she found a hole in front of them. 

The fox smiled and followed the ants while they were about. Soon the wind shone high on the sides and caused the safety. Inside, it was nothing so bad at the hospital and it was an satisfied. The little girl was so content that, it started to get happy. 

The little girl returned and thanked the hunter for the rest of the day. Every day every day, she went back to the forest and met lots of new friends. The end.Once upon a time, there was a little girl. She loved to play a sunshine, especially afternoon because she smiled happily and started to make bouncing her feel brave. 

One day, a princess came to her house. She made her like the flowers and it hers. But the princess was very unlucky. She felt
