import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

from model import SouthIndianCaptionModel, SOUTH_INDIAN_LANGUAGES
from dataset import build_dataloader

# ── Config ────────────────────────────────────────────────────
EPOCHS      = 15
BATCH_SIZE  = 8
LR          = 3e-4
CLIP_W      = 0.3
EMOTION_W   = 0.2
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT  = "south_indian_caption_model.pt"
DATA_JSON   = Path("data/south_indian_captions.json")

# ── Load data ─────────────────────────────────────────────────
if DATA_JSON.exists():
    with open(DATA_JSON, encoding="utf-8") as f:
        all_data = json.load(f)
    split      = int(len(all_data) * 0.9)
    TRAIN_DATA = all_data[:split]
    print(f"Loaded {len(TRAIN_DATA)} training samples")
else:
    print("No dataset found. Run: python prepare_data.py")
    TRAIN_DATA = []


def contrastive_loss(img_feats, txt_feats):
    # pool both to (batch, dim) — project txt to same dim as img if needed
    img = F.normalize(img_feats.mean(1), dim=-1)          # (B, 1024)
    txt = F.normalize(txt_feats.mean(1), dim=-1)          # (B, 1024)
    if img.shape[-1] != txt.shape[-1]:
        return torch.tensor(0.0, device=img.device)       # skip if dims mismatch
    logits = img @ txt.T * 10.0
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train():
    if not TRAIN_DATA:
        print("No training data. Run: python prepare_data.py first.")
        return

    print(f"Device: {DEVICE}")
    model  = SouthIndianCaptionModel().to(DEVICE)
    loader = build_dataloader(TRAIN_DATA, batch_size=BATCH_SIZE)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M")

    optimizer = AdamW(trainable, lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    emo_loss_fn = torch.nn.CrossEntropyLoss()
    best_loss   = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        lang_losses = defaultdict(list)
        total_loss  = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            pixel_values   = batch["pixel_values"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            lang_names     = batch["lang_names"]
            emotion_labels = batch["emotion_labels"].to(DEVICE)

            loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

            for lang in set(lang_names):
                mask = [i for i, l in enumerate(lang_names) if l == lang]
                pv   = pixel_values[mask]
                lb   = labels[mask]
                el   = emotion_labels[mask]

                out, emo_logits = model(pixel_values=pv, labels=lb, lang_name=lang)

                cap_loss = out.loss
                emo_loss = emo_loss_fn(emo_logits, el)

                # contrastive: MoE features vs script-pos shifted features (same dim=1024)
                vis_out    = model.encoder(pixel_values=pv)
                img_feats  = model.moe_proj(vis_out.last_hidden_state)
                lang_idx   = model.lang_list.index(lang)
                pos_feats  = model.script_pos(img_feats, lang_idx=lang_idx)
                clip_loss  = contrastive_loss(img_feats, pos_feats)

                lang_loss = cap_loss + CLIP_W * clip_loss + EMOTION_W * emo_loss
                loss      = loss + lang_loss / len(set(lang_names))
                lang_losses[lang].append(cap_loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        print(f"\nEpoch {epoch+1} | Loss: {avg:.4f}")
        for lang, losses in lang_losses.items():
            print(f"  {lang:10s}: {sum(losses)/len(losses):.4f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ Saved best model (loss={best_loss:.4f})")


if __name__ == "__main__":
    train()
