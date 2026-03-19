#!/usr/bin/env python3
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

print("Downloading DistilGPT-2...")
model = AutoModelForCausalLM.from_pretrained('distilgpt2')
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token

print("Splitting model into network layers...")

# Stage 1: Embeddings + first 2 transformer blocks
stage1 = nn.Sequential(
    model.transformer.wte,
    model.transformer.wpe,
    model.transformer.drop,
    *model.transformer.h[:2]
)

# Stage 2: Middle 2 blocks
stage2 = nn.Sequential(*model.transformer.h[2:4])

# Stage 3: Last 2 blocks + head
stage3 = nn.Sequential(
    *model.transformer.h[4:],
    model.transformer.ln_f,
    model.lm_head
)

# Save
torch.save(stage1, 'chat_layer1.pt')
torch.save(stage2, 'chat_layer2.pt')
torch.save(stage3, 'chat_layer3.pt')
tokenizer.save_pretrained('./chat_tokenizer')

print("Model prepared successfully!")
print("Files: chat_layer1.pt, chat_layer2.pt, chat_layer3.pt")
