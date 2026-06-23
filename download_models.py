import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
from PIL import Image
from transformers import (
    CLIPVisionModel, CLIPProcessor,
    MarianMTModel, MarianTokenizer,
    BertTokenizer,
)

print("=" * 55)
print("  Downloading pretrained weights (lightweight stack)")
print("=" * 55)

print("\n[1] CLIP ViT-B/32  (~340 MB) ...")
clip  = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(f"    OK  hidden={clip.config.hidden_size}  "
      f"params={sum(p.numel() for p in clip.parameters())/1e6:.1f}M")

print("\n[2] MarianMT en→Dravidian  (~300 MB) ...")
marian_tok   = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-dra")
marian_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-dra")
print(f"    OK  d_model={marian_model.config.d_model}  "
      f"params={sum(p.numel() for p in marian_model.parameters())/1e6:.1f}M")

print("\n[3] BERT tokenizer for English head  (~500 KB) ...")
bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
print(f"    OK  vocab_size={bert_tok.vocab_size}")

print("\n[4] Checking South Indian language tokens in MarianMT ...")
si_prefixes = {"Tamil":">>ta<<","Telugu":">>te<<","Kannada":">>kn<<","Malayalam":">>ml<<"}
for lang, prefix in si_prefixes.items():
    tid = marian_tok.convert_tokens_to_ids(prefix)
    print(f"    {lang:10s}  {prefix}  →  token_id={tid}")

print("\n[5] Smoke test — CLIP forward pass ...")
dummy_img  = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
pv         = cproc(images=dummy_img, return_tensors="pt").pixel_values
with torch.no_grad():
    out = clip(pixel_values=pv)
print(f"    OK  image features: {out.last_hidden_state.shape}")

print("\n[6] Smoke test — MarianMT translation ...")
src = [">>ta<< A beautiful temple in South India"]
tok = marian_tok(src, return_tensors="pt", padding=True)
with torch.no_grad():
    ids = marian_model.generate(**tok, max_length=50)
result = marian_tok.batch_decode(ids, skip_special_tokens=True)
print(f"    Input : {src[0]}")
print(f"    Output: {result[0]}")

print("\n" + "=" * 55)
print("  All weights downloaded!")
print("  Next: python prepare_data.py  →  python train.py")
print("=" * 55)
