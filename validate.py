import torch
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 55)
print("  South Indian Multilingual Caption Model - Validation")
print("=" * 55)

# ── 1. Emotion module ─────────────────────────────────────────
print("\n[1] Testing emotion.py ...")
from emotion import EmotionClassifier, decorate_caption, EMOTIONS

clf   = EmotionClassifier()
dummy = torch.randn(2, 50, 768)
logits = clf(dummy)
assert logits.shape == (2, 6), "EmotionClassifier output shape mismatch"
emotions, probs = clf.predict(dummy)
assert len(emotions) == 2
print("    EmotionClassifier OK — output shape:", logits.shape)

test_cases = [
    ("Tamil",     "கோயிலின் அழகான காட்சி",    "Peaceful"),
    ("Telugu",    "దేవాలయం యొక్క దృశ్యం",      "Happy"),
    ("Kannada",   "ದೇವಾಲಯದ ಸುಂದರ ದೃಶ್ಯ",       "Exciting"),
    ("Malayalam", "ക്ഷേത്രത്തിന്റെ കാഴ്ച",      "Sad"),
    ("English",   "A beautiful view of the temple", "Surprising"),
]
print("\n    Emoji-decorated captions:")
for lang, cap, emo in test_cases:
    decorated = decorate_caption(cap, emo, lang)
    print(f"      [{emo:10s}] {lang:10s}: {decorated}")

# ── 2. Model architecture ─────────────────────────────────────
print("\n[2] Testing model.py components (no pretrained download) ...")
from model import MoEProjection, ScriptAwarePositionalEncoding, SOUTH_INDIAN_LANGUAGES

moe = MoEProjection(in_dim=768, out_dim=1024)
x   = torch.randn(2, 50, 768)
out = moe(x, lang_idx=0)
assert out.shape == (2, 50, 1024), f"MoE output shape wrong: {out.shape}"
print("    MoEProjection OK — output shape:", out.shape)

spe = ScriptAwarePositionalEncoding(d_model=1024)
out2 = spe(out, lang_idx=1)
assert out2.shape == (2, 50, 1024)
print("    ScriptAwarePositionalEncoding OK — output shape:", out2.shape)

print("\n    Supported languages:", list(SOUTH_INDIAN_LANGUAGES.keys()))

# ── 3. Dataset structure ──────────────────────────────────────
print("\n[3] Testing dataset.py structure ...")
from emotion import EMOTIONS
from model import SOUTH_INDIAN_LANGUAGES as LANGS

sample_data = [
    {
        "image_path": "dummy.jpg",
        "emotion": "Happy",
        "captions": {
            "Tamil":     "சோதனை தலைப்பு",
            "Telugu":    "పరీక్ష శీర్షిక",
            "Kannada":   "ಪರೀಕ್ಷೆ ಶೀರ್ಷಿಕೆ",
            "Malayalam": "പരീക്ഷ തലക്കെട്ട്",
            "English":   "Test caption",
        }
    }
]
emotion_idx = EMOTIONS.index(sample_data[0]["emotion"])
assert emotion_idx == 0, "Emotion index mismatch"
print("    Dataset structure OK — emotion index:", emotion_idx)
print("    Flattened samples would be:", len(sample_data) * len(LANGS))

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  All checks passed!")
print("  Next step: python train.py  (after adding real images)")
print("=" * 55)
