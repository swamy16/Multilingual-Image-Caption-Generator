import torch
import torch.nn as nn
import torch.nn.functional as F


EMOTIONS = ["Happy", "Sad", "Peaceful", "Exciting", "Fearful", "Surprising"]

# Emoji + symbol decorators per emotion, per language feel
EMOTION_EMOJI_MAP = {
    "Happy": {
        "Tamil":     ["😊", "🌟", "✨", "🎉", "💛"],
        "Telugu":    ["😄", "🌸", "🎊", "💫", "🌼"],
        "Kannada":   ["😁", "🌺", "🎈", "⭐", "🌻"],
        "Malayalam": ["🥰", "🌷", "💐", "🌈", "✨"],
        "English":   ["😊", "🌟", "🎉", "✨", "💛"],
    },
    "Sad": {
        "Tamil":     ["😢", "🌧️", "💔", "🥀", "😞"],
        "Telugu":    ["😭", "🌨️", "💧", "🥺", "😔"],
        "Kannada":   ["😿", "⛈️", "💦", "😓", "🌑"],
        "Malayalam": ["😥", "🌩️", "🖤", "😪", "🌫️"],
        "English":   ["😢", "💔", "🌧️", "🥺", "😞"],
    },
    "Peaceful": {
        "Tamil":     ["😌", "🕊️", "🌿", "🍃", "☮️"],
        "Telugu":    ["🧘", "🌾", "🍀", "🌊", "🕊️"],
        "Kannada":   ["😇", "🌅", "🌄", "🍃", "🌙"],
        "Malayalam": ["🙏", "🌸", "🌿", "☁️", "🕊️"],
        "English":   ["😌", "🕊️", "🌿", "☮️", "🍃"],
    },
    "Exciting": {
        "Tamil":     ["🤩", "🔥", "⚡", "🚀", "💥"],
        "Telugu":    ["😍", "🎯", "🏆", "⚡", "🔥"],
        "Kannada":   ["🥳", "🎆", "🎇", "💥", "🚀"],
        "Malayalam": ["😲", "🎉", "🎊", "⚡", "🌟"],
        "English":   ["🤩", "🔥", "⚡", "🚀", "💥"],
    },
    "Fearful": {
        "Tamil":     ["😨", "👻", "🌑", "⚠️", "😰"],
        "Telugu":    ["😱", "🕷️", "🌚", "⛔", "😖"],
        "Kannada":   ["😧", "🦇", "🌒", "❗", "😟"],
        "Malayalam": ["😦", "👁️", "🌫️", "⚡", "😬"],
        "English":   ["😨", "👻", "⚠️", "🌑", "😰"],
    },
    "Surprising": {
        "Tamil":     ["😮", "🎭", "❓", "🌀", "✨"],
        "Telugu":    ["😯", "🎪", "❗", "💫", "🌀"],
        "Kannada":   ["😲", "🎠", "‼️", "⁉️", "🌟"],
        "Malayalam": ["🤯", "🎡", "❓", "💥", "✨"],
        "English":   ["😮", "🎭", "❓", "🌀", "✨"],
    },
}


class EmotionClassifier(nn.Module):
    """
    Lightweight MLP on top of CLIP image features.
    Predicts one of 6 emotions from the image.
    """
    def __init__(self, in_dim: int = 768, num_emotions: int = len(EMOTIONS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, num_emotions),
        )

    def forward(self, image_features):
        # image_features: (batch, seq, dim) → pool → (batch, dim)
        pooled = image_features.mean(dim=1)
        return self.net(pooled)   # (batch, num_emotions)

    def predict(self, image_features):
        logits = self.forward(image_features)
        probs  = F.softmax(logits, dim=-1)
        idx    = probs.argmax(dim=-1)
        return [EMOTIONS[i] for i in idx], probs


def decorate_caption(caption: str, emotion: str, language: str, num_emojis: int = 2) -> str:
    """
    Wraps a caption with emotion-appropriate emojis.
    Picks the top `num_emojis` symbols for the language.
    """
    emojis = EMOTION_EMOJI_MAP.get(emotion, {}).get(language, ["✨", "🌟"])
    prefix = " ".join(emojis[:num_emojis])
    suffix = " ".join(emojis[num_emojis:num_emojis + 1])
    return f"{prefix}  {caption}  {suffix}"
