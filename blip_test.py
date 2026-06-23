import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from PIL import Image
from pathlib import Path
from pipeline import SouthIndianCaptioner
from emotion import EMOTIONS

print("=" * 60)
print("  South Indian Caption Generator — Pipeline Test")
print("=" * 60)

captioner = SouthIndianCaptioner()

# Use first image from prepared dataset if available, else create dummy
img_path = Path("data/images/img_00000.jpg")
if img_path.exists():
    image = Image.open(img_path).convert("RGB")
    print(f"\nUsing real image: {img_path}")
else:
    import numpy as np
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    print("\nUsing dummy image (no dataset found)")

result = captioner.caption(image)

print(f"\n  BLIP English : {result['english_raw']}")
print(f"  Emotion      : {result['emotion']}")

print("\n  Emotion Confidence:")
for emo, prob in zip(EMOTIONS, result["probs"]):
    bar = "█" * int(prob.item() * 30)
    print(f"    {emo:12s} {bar:<30s} {prob.item()*100:.1f}%")

print("\n  Captions with Emojis:")
print("  " + "-" * 56)
for lang, cap in result["captions"].items():
    print(f"  {lang:10s} │ {cap}")

print("\n" + "=" * 60)
print("  Pipeline working! Run: python -X utf8 app.py")
print("=" * 60)
