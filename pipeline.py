import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    MarianMTModel, MarianTokenizer,
    CLIPProcessor, CLIPModel,
)
from emotion import EmotionClassifier, decorate_caption, EMOTIONS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SOUTH_INDIAN_LANGUAGES = {
    "Tamil":     ">>tam<<",
    "Telugu":    ">>tel<<",
    "Kannada":   ">>kan<<",
    "Malayalam": ">>mal<<",
    "English":   None,
}

MARIAN_MODEL = "Helsinki-NLP/opus-mt-en-mul"


class SouthIndianCaptioner:
    """
    Pipeline:
      Image → BLIP (English caption)
            → MarianMT (translate to 4 South Indian languages)
            → CLIP features → EmotionClassifier (detect emotion)
            → decorate_caption (add emojis per language + emotion)
    """

    def __init__(self):
        print("Loading BLIP captioning model...")
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(DEVICE)
        self.blip_model.eval()

        print("Loading MarianMT translation model...")
        self.marian_tok   = MarianTokenizer.from_pretrained(MARIAN_MODEL)
        self.marian_model = MarianMTModel.from_pretrained(MARIAN_MODEL).to(DEVICE)
        self.marian_model.eval()

        print("Loading CLIP for emotion detection...")
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(DEVICE)
        self.clip_model.eval()

        self.emotion_clf = EmotionClassifier(in_dim=768).to(DEVICE)
        self.emotion_clf.eval()

        print(f"All models loaded on {DEVICE}!")

    @torch.no_grad()
    def _generate_english(self, image: Image.Image) -> str:
        inputs = self.blip_processor(images=image, return_tensors="pt").to(DEVICE)
        ids    = self.blip_model.generate(
            **inputs,
            max_length       = 64,
            num_beams        = 4,
            no_repeat_ngram_size = 3,
        )
        return self.blip_processor.decode(ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def _translate(self, text: str, lang_prefix: str) -> str:
        src = [f"{lang_prefix} {text}"]
        tok = self.marian_tok(src, return_tensors="pt",
                              padding=True, truncation=True,
                              max_length=128).to(DEVICE)
        ids = self.marian_model.generate(**tok, max_length=128, num_beams=4)
        return self.marian_tok.decode(ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def _detect_emotion(self, image: Image.Image):
        inputs = self.clip_processor(images=image, return_tensors="pt").to(DEVICE)
        feats  = self.clip_model.vision_model(**inputs).last_hidden_state
        emotions, probs = self.emotion_clf.predict(feats)
        return emotions[0], probs[0]

    def caption(self, image: Image.Image) -> dict:
        """
        Returns:
          {
            "english_raw": str,
            "emotion":     str,
            "probs":       tensor,
            "captions": {
                "Tamil":     "emoji caption emoji",
                "Telugu":    "...",
                "Kannada":   "...",
                "Malayalam": "...",
                "English":   "...",
            }
          }
        """
        image = image.convert("RGB")

        # Step 1 — Generate English caption with BLIP
        english = self._generate_english(image)

        # Step 2 — Detect emotion from image
        emotion, probs = self._detect_emotion(image)

        # Step 3 — Translate + decorate per language
        captions = {}
        for lang_name, prefix in SOUTH_INDIAN_LANGUAGES.items():
            if prefix is None:
                text = english
            else:
                text = self._translate(english, prefix)
            captions[lang_name] = decorate_caption(text, emotion, lang_name)

        return {
            "english_raw": english,
            "emotion":     emotion,
            "probs":       probs,
            "captions":    captions,
        }
