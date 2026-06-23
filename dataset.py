import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, MarianTokenizer
from PIL import Image
from model import SOUTH_INDIAN_LANGUAGES, MARIAN_LANG_PREFIX
from emotion import EMOTIONS

MARIAN_MODEL = "Helsinki-NLP/opus-mt-en-mul"


class SouthIndianCaptionDataset(Dataset):
    """
    data format:
    [
      {
        "image_path": "data/images/img_00001.jpg",
        "emotion": "Happy",
        "captions": {
            "Tamil":     "...",
            "Telugu":    "...",
            "Kannada":   "...",
            "Malayalam": "...",
            "English":   "..."
        }
      }, ...
    ]
    """
    def __init__(self, data, clip_model="openai/clip-vit-base-patch32", max_length=64):
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.marian_tok     = MarianTokenizer.from_pretrained(MARIAN_MODEL)
        self.max_length     = max_length

        self.samples = []
        for item in data:
            emotion_idx = EMOTIONS.index(item.get("emotion", "Peaceful"))
            for lang_name in SOUTH_INDIAN_LANGUAGES:
                if lang_name in item["captions"]:
                    self.samples.append({
                        "image_path":  item["image_path"],
                        "caption":     item["captions"][lang_name],
                        "lang_name":   lang_name,
                        "emotion_idx": emotion_idx,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        pixel_values = self.clip_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        lang_name = s["lang_name"]
        caption   = s["caption"]

        if lang_name == "English":
            from transformers import BertTokenizer
            en_tok = BertTokenizer.from_pretrained("bert-base-uncased")
            tok    = en_tok(caption, max_length=self.max_length,
                            padding="max_length", truncation=True, return_tensors="pt")
        else:
            prefix  = MARIAN_LANG_PREFIX[lang_name]
            prefixed = f"{prefix} {caption}"
            tok = self.marian_tok(prefixed, max_length=self.max_length,
                                  padding="max_length", truncation=True, return_tensors="pt")

        labels = tok.input_ids.squeeze(0).clone()
        pad_id = (self.marian_tok.pad_token_id
                  if lang_name != "English" else 0)
        labels[labels == pad_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels":        labels,
            "lang_name":     lang_name,
            "emotion_label": torch.tensor(s["emotion_idx"], dtype=torch.long),
        }


def collate_fn(batch):
    return {
        "pixel_values":  torch.stack([b["pixel_values"] for b in batch]),
        "labels":         torch.stack([b["labels"] for b in batch]),
        "lang_names":    [b["lang_name"] for b in batch],
        "emotion_labels": torch.stack([b["emotion_label"] for b in batch]),
    }


def build_dataloader(data, batch_size=8, shuffle=True):
    dataset = SouthIndianCaptionDataset(data)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_fn)
