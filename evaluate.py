"""
evaluate.py
Runs BLEU + CIDEr evaluation on the test split,
broken down per language and per emotion.
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import CLIPProcessor, MBart50TokenizerFast
from sacrebleu.metrics import BLEU
from model import SouthIndianCaptionModel, SOUTH_INDIAN_LANGUAGES
from emotion import EMOTIONS

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT  = "south_indian_caption_model.pt"
DATA_JSON   = "data/south_indian_captions.json"
TEST_SPLIT  = 0.1   # last 10% used as test set


def load_artifacts():
    model = SouthIndianCaptionModel().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer      = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
    return model, clip_processor, tokenizer


def evaluate():
    from PIL import Image

    with open(DATA_JSON, encoding="utf-8") as f:
        data = json.load(f)

    split_idx = int(len(data) * (1 - TEST_SPLIT))
    test_data = data[split_idx:]
    print(f"Evaluating on {len(test_data)} test samples...")

    model, clip_processor, tokenizer = load_artifacts()
    bleu = BLEU(effective_order=True)

    # predictions[lang][emotion] = (preds, refs)
    preds_by_lang  = defaultdict(list)
    refs_by_lang   = defaultdict(list)
    preds_by_emotion = defaultdict(list)
    refs_by_emotion  = defaultdict(list)

    emotion_correct = 0
    emotion_total   = 0

    for item in tqdm(test_data, desc="Evaluating"):
        image        = Image.open(item["image_path"]).convert("RGB")
        pixel_values = clip_processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
        true_emotion = item["emotion"]

        for lang_name, lang_code in SOUTH_INDIAN_LANGUAGES.items():
            if lang_name not in item["captions"]:
                continue

            with torch.no_grad():
                captions, pred_emotion, _ = model.generate_caption(
                    pixel_values, tokenizer,
                    lang_code=lang_code, language_name=lang_name
                )

            # Strip emojis for BLEU (compare text only)
            pred_text = captions[0].split("  ")[1] if "  " in captions[0] else captions[0]
            ref_text  = item["captions"][lang_name]

            preds_by_lang[lang_name].append(pred_text)
            refs_by_lang[lang_name].append(ref_text)
            preds_by_emotion[true_emotion].append(pred_text)
            refs_by_emotion[true_emotion].append(ref_text)

            if lang_name == "English":
                emotion_correct += int(pred_emotion == true_emotion)
                emotion_total   += 1

    # ── Print results ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  BLEU Scores by Language")
    print("=" * 55)
    for lang in SOUTH_INDIAN_LANGUAGES:
        if preds_by_lang[lang]:
            score = bleu.corpus_score(preds_by_lang[lang], [refs_by_lang[lang]])
            bar   = "█" * int(score.score / 5)
            print(f"  {lang:10s} │ {bar:<20s} {score.score:.2f}")

    print("\n" + "=" * 55)
    print("  BLEU Scores by Emotion")
    print("=" * 55)
    for emo in EMOTIONS:
        if preds_by_emotion[emo]:
            score = bleu.corpus_score(preds_by_emotion[emo], [refs_by_emotion[emo]])
            bar   = "█" * int(score.score / 5)
            print(f"  {emo:12s} │ {bar:<20s} {score.score:.2f}")

    if emotion_total > 0:
        acc = emotion_correct / emotion_total * 100
        print(f"\n  Emotion Classification Accuracy: {acc:.1f}%")

    print("=" * 55)


if __name__ == "__main__":
    evaluate()
