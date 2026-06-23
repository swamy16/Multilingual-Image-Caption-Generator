import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, MarianTokenizer

print("=" * 60)
print("  Full Model Build & Forward Pass Test")
print("=" * 60)

# ── Build model ───────────────────────────────────────────────
print("\n[1] Building SouthIndianCaptionModel ...")
from model import SouthIndianCaptionModel, SOUTH_INDIAN_LANGUAGES, MARIAN_LANG_PREFIX, MARIAN_MODEL
model = SouthIndianCaptionModel()
model.eval()

total  = sum(p.numel() for p in model.parameters()) / 1e6
train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
frozen = total - train
print(f"    Total params  : {total:.1f}M")
print(f"    Trainable     : {train:.1f}M  (LoRA + MoE + ScriptPos + EmotionClf + EnHead)")
print(f"    Frozen (CLIP) : {frozen:.1f}M")

# ── Prepare dummy image ───────────────────────────────────────
print("\n[2] Preparing dummy image ...")
clip_proc    = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
marian_tok   = MarianTokenizer.from_pretrained(MARIAN_MODEL)
dummy_img    = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
pixel_values = clip_proc(images=dummy_img, return_tensors="pt").pixel_values
print(f"    pixel_values shape: {pixel_values.shape}")

# ── Forward pass for each language ───────────────────────────
print("\n[3] Forward pass per language ...")
from transformers import BertTokenizer
bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")

for lang_name in SOUTH_INDIAN_LANGUAGES:
    if lang_name == "English":
        dummy_caption = "A beautiful temple in South India"
        tok = bert_tok(dummy_caption, return_tensors="pt",
                       max_length=32, padding="max_length", truncation=True)
        labels = tok.input_ids.clone()
        labels[labels == 0] = -100
    else:
        prefix = MARIAN_LANG_PREFIX[lang_name]
        dummy_caption = f"{prefix} A beautiful temple in South India"
        tok = marian_tok(dummy_caption, return_tensors="pt",
                         max_length=32, padding="max_length", truncation=True)
        labels = tok.input_ids.clone()
        labels[labels == marian_tok.pad_token_id] = -100

    with torch.no_grad():
        out, emo_logits = model(pixel_values=pixel_values, labels=labels, lang_name=lang_name)

    print(f"    {lang_name:10s}  loss={out.loss.item():.4f}  "
          f"logits={out.logits.shape}  emo={emo_logits.shape}")

# ── Generate captions ─────────────────────────────────────────
print("\n[4] Generating captions (untrained — random output expected) ...")
for lang_name in SOUTH_INDIAN_LANGUAGES:
    with torch.no_grad():
        captions, emotion, probs = model.generate_caption(
            pixel_values, marian_tok, lang_name=lang_name
        )
    top_emo = f"{emotion} ({probs.max().item()*100:.0f}%)"
    print(f"    {lang_name:10s} [{top_emo:20s}]: {captions[0]}")

print("\n" + "=" * 60)
print("  Model build complete! Ready to train.")
print("  Run: python prepare_data.py  then  python train.py")
print("=" * 60)
