import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json, torch
from pathlib import Path
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset, concatenate_datasets
from collections import Counter

DATA_DIR      = Path("data")
IMAGE_DIR     = DATA_DIR / "images"
OUTPUT_JSON   = DATA_DIR / "south_indian_captions.json"
PROGRESS_JSON = DATA_DIR / "translation_progress.json"   # checkpoint file
MARIAN_MODEL  = "Helsinki-NLP/opus-mt-en-mul"
BATCH_SIZE    = 64

LANG_PREFIXES = {
    "Tamil":     ">>tam<<",
    "Telugu":    ">>tel<<",
    "Kannada":   ">>kan<<",
    "Malayalam": ">>mal<<",
}

EMOTION_KEYWORDS = {
    "Happy":      ["smile","laugh","happy","joy","celebrat","play","fun","cheer","enjoy","festival","danc"],
    "Sad":        ["cry","sad","alone","dark","rain","tear","grief","lonely","mourn","weep"],
    "Peaceful":   ["calm","quiet","nature","temple","garden","sunset","serene","rest","meditat","pray","sit"],
    "Exciting":   ["run","jump","race","sport","crowd","fast","action","fire","compet","kick","throw","catch"],
    "Fearful":    ["dark","storm","danger","scary","night","shadow","threat","wild","fierce","growl"],
    "Surprising": ["surprise","shock","unexpected","sudden","wow","amazing","strange","unusual","stare"],
}

def detect_emotion(caption: str) -> str:
    c = caption.lower()
    scores = {e: sum(kw in c for kw in kws) for e, kws in EMOTION_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Peaceful"

def translate_batch(texts, prefix, tokenizer, model, device, max_len=80):
    prefixed = [f"{prefix} {t}" for t in texts]
    tok = tokenizer(prefixed, return_tensors="pt", padding=True,
                    truncation=True, max_length=128).to(device)
    with torch.no_grad():
        ids = model.generate(**tok, max_length=max_len, num_beams=2)
    return tokenizer.batch_decode(ids, skip_special_tokens=True)

def prepare():
    DATA_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load checkpoint if exists ─────────────────────────────
    if PROGRESS_JSON.exists():
        with open(PROGRESS_JSON, encoding="utf-8") as f:
            progress = json.load(f)
        print(f"Resuming from checkpoint — already done: {list(progress['translations'].keys())}")
    else:
        progress = {"translations": {}}

    translations = progress["translations"]

    # ── Load dataset ──────────────────────────────────────────
    print("\nLoading Flickr8k (train + test)...")
    train_ds = load_dataset("jxie/flickr8k", split="train")
    test_ds  = load_dataset("jxie/flickr8k", split="test")
    dataset  = concatenate_datasets([train_ds, test_ds])
    print(f"Total samples: {len(dataset)}")

    english_captions = [item["caption_0"] for item in dataset]

    if "English" not in translations:
        translations["English"] = english_captions
        _save_progress(progress)

    # ── Load MarianMT ─────────────────────────────────────────
    print("\nLoading MarianMT (en→mul)...")
    tokenizer = MarianTokenizer.from_pretrained(MARIAN_MODEL)
    mt_model  = MarianMTModel.from_pretrained(MARIAN_MODEL).to(device)
    mt_model.eval()

    # ── Translate each language (skip if already done) ────────
    for lang_name, prefix in LANG_PREFIXES.items():
        if lang_name in translations:
            print(f"  ✓ {lang_name} already done — skipping")
            continue

        print(f"\n  → Translating {lang_name} ...")
        lang_trans = []
        for i in tqdm(range(0, len(english_captions), BATCH_SIZE), desc=f"    {lang_name}"):
            batch = english_captions[i: i + BATCH_SIZE]
            lang_trans.extend(translate_batch(batch, prefix, tokenizer, mt_model, device))

        translations[lang_name] = lang_trans
        _save_progress(progress)   # save after every language
        print(f"    Sample: {lang_trans[0]}")

    # ── Save images + build final JSON ────────────────────────
    print("\nSaving images and building JSON...")
    results = []
    for idx, item in enumerate(tqdm(dataset, desc="Saving")):
        img_path = IMAGE_DIR / f"img_{idx:05d}.jpg"
        if not img_path.exists():
            item["image"].save(str(img_path))

        emotion = detect_emotion(english_captions[idx])
        results.append({
            "image_path": str(img_path),
            "emotion":    emotion,
            "captions": {
                "Tamil":     translations["Tamil"][idx],
                "Telugu":    translations["Telugu"][idx],
                "Kannada":   translations["Kannada"][idx],
                "Malayalam": translations["Malayalam"][idx],
                "English":   english_captions[idx],
            }
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Clean up checkpoint
    PROGRESS_JSON.unlink(missing_ok=True)

    print(f"\n✓ Saved {len(results)} samples → {OUTPUT_JSON}")

    dist = Counter(r["emotion"] for r in results)
    print("\nEmotion distribution:")
    for emo, cnt in dist.most_common():
        bar = "█" * (cnt // 30)
        print(f"  {emo:12s} {bar} {cnt}")

    print("\nSample entry:")
    s = results[0]
    print(f"  image  : {s['image_path']}")
    print(f"  emotion: {s['emotion']}")
    for lang, cap in s["captions"].items():
        print(f"  {lang:10s}: {cap}")

def _save_progress(progress: dict):
    with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)

if __name__ == "__main__":
    prepare()
