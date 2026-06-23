import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, MarianMTModel, MarianTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from emotion import EmotionClassifier, decorate_caption, EMOTIONS

SOUTH_INDIAN_LANGUAGES = {
    "Tamil":     "ta",
    "Telugu":    "te",
    "Kannada":   "kn",
    "Malayalam": "ml",
    "English":   "en",
}

MARIAN_MODEL = "Helsinki-NLP/opus-mt-en-mul"

# Verified language prefixes (token IDs confirmed)
MARIAN_LANG_PREFIX = {
    "Tamil":     ">>tam<<",
    "Telugu":    ">>tel<<",
    "Kannada":   ">>kan<<",
    "Malayalam": ">>mal<<",
    "English":   None,
}

NUM_LANGUAGES = len(SOUTH_INDIAN_LANGUAGES)


# ── 1. Mixture of Experts Projection ─────────────────────────────────────────
class MoEProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_experts: int = NUM_LANGUAGES):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_experts)])
        self.gate    = nn.Linear(in_dim, num_experts)

    def forward(self, x, lang_idx: int = None):
        gate_logits = self.gate(x.mean(dim=1))
        if lang_idx is not None:
            weights = torch.zeros_like(gate_logits)
            weights[:, lang_idx] = 1.0
        else:
            weights = F.softmax(gate_logits, dim=-1)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        return (weights * expert_outs).sum(dim=1)


# ── 2. Script-Aware Positional Encoding ──────────────────────────────────────
class ScriptAwarePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128, num_languages: int = NUM_LANGUAGES):
        super().__init__()
        self.lang_pos_bias = nn.Embedding(num_languages * max_len, d_model)
        self.max_len = max_len

    def forward(self, x, lang_idx: int):
        seq_len   = x.size(1)
        positions = torch.arange(seq_len, device=x.device) + lang_idx * self.max_len
        return x + self.lang_pos_bias(positions).unsqueeze(0)


# ── 3. English Caption Head (lightweight decoder on CLIP features) ────────────
class EnglishCaptionHead(nn.Module):
    """Small transformer decoder that generates English captions from CLIP features."""
    def __init__(self, d_model: int = 512, vocab_size: int = 30522,
                 nhead: int = 8, num_layers: int = 4, max_len: int = 64):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        decoder_layer  = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048,
                                                     dropout=0.1, batch_first=True)
        self.decoder   = nn.TransformerDecoder(decoder_layer, num_layers)
        self.proj_in   = nn.Linear(1024, d_model)   # from MoE output dim
        self.out       = nn.Linear(d_model, vocab_size)
        self.max_len   = max_len

    def forward(self, memory, tgt_ids):
        B, S = tgt_ids.shape
        pos  = torch.arange(S, device=tgt_ids.device).unsqueeze(0)
        tgt  = self.embed(tgt_ids) + self.pos_embed(pos)
        mem  = self.proj_in(memory)
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=tgt_ids.device)
        out  = self.decoder(tgt, mem, tgt_mask=mask)
        return self.out(out)

    @torch.no_grad()
    def generate(self, memory, tokenizer, max_length: int = 64):
        B   = memory.size(0)
        mem = self.proj_in(memory)
        ids = torch.full((B, 1), tokenizer.cls_token_id or 101,
                         dtype=torch.long, device=memory.device)
        for _ in range(max_length - 1):
            pos  = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
            tgt  = self.embed(ids) + self.pos_embed(pos)
            mask = nn.Transformer.generate_square_subsequent_mask(
                ids.size(1), device=ids.device)
            out  = self.decoder(tgt, mem, tgt_mask=mask)
            next_id = self.out(out[:, -1, :]).argmax(-1, keepdim=True)
            ids  = torch.cat([ids, next_id], dim=1)
            if (next_id == (tokenizer.sep_token_id or 102)).all():
                break
        return tokenizer.batch_decode(ids, skip_special_tokens=True)


# ── 4. Main Model ─────────────────────────────────────────────────────────────
class SouthIndianCaptionModel(nn.Module):
    def __init__(self, clip_model="openai/clip-vit-base-patch32", lora_r=8, lora_alpha=16):
        super().__init__()

        # CLIP encoder — frozen
        self.encoder = CLIPVisionModel.from_pretrained(clip_model)
        for p in self.encoder.parameters():
            p.requires_grad = False

        clip_dim  = self.encoder.config.hidden_size   # 768
        moe_dim   = 1024

        self.moe_proj   = MoEProjection(clip_dim, moe_dim)
        self.script_pos = ScriptAwarePositionalEncoding(moe_dim)
        self.emotion_clf = EmotionClassifier(in_dim=clip_dim)

        # MarianMT multilingual model — supports all 4 South Indian languages
        print("Loading MarianMT (en→mul)...")
        base_marian = MarianMTModel.from_pretrained(MARIAN_MODEL)
        lora_cfg = LoraConfig(
            task_type      = TaskType.SEQ_2_SEQ_LM,
            r              = lora_r,
            lora_alpha     = lora_alpha,
            lora_dropout   = 0.1,
            target_modules = ["q_proj", "v_proj"],
        )
        self.marian    = get_peft_model(base_marian, lora_cfg)
        self.marian_d  = base_marian.config.d_model   # 512

        # Project MoE output → MarianMT d_model
        self.moe_to_marian = nn.Linear(moe_dim, self.marian_d)

        # English caption head
        self.en_head = EnglishCaptionHead(d_model=512, vocab_size=30522)

        self.lang_list = list(SOUTH_INDIAN_LANGUAGES.keys())

    def _lang_idx(self, lang_name: str) -> int:
        return self.lang_list.index(lang_name)

    def forward(self, pixel_values, labels, lang_name: str):
        lang_idx = self._lang_idx(lang_name)

        vision_out   = self.encoder(pixel_values=pixel_values)
        raw_features = vision_out.last_hidden_state          # (B, 50, 768)

        emotion_logits = self.emotion_clf(raw_features)

        image_features = self.moe_proj(raw_features, lang_idx=lang_idx)   # (B, 50, 1024)
        image_features = self.script_pos(image_features, lang_idx=lang_idx)

        if lang_name == "English":
            # Clamp labels to valid range before passing to en_head (ignore -100 padding)
            tgt_ids = labels[:, :-1].clone()
            tgt_ids = tgt_ids.clamp(min=0)
            logits = self.en_head(image_features, tgt_ids)
            shift_labels = labels[:, 1:].clone()
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100
            )
            return type('Out', (), {'loss': loss, 'logits': logits})(), emotion_logits

        marian_features = self.moe_to_marian(image_features)   # (B, 50, 512)
        out = self.marian(inputs_embeds=marian_features, labels=labels)
        return out, emotion_logits

    @torch.no_grad()
    def generate_caption(self, pixel_values, tokenizer, lang_name: str, max_length: int = 64):
        lang_idx = self._lang_idx(lang_name)

        vision_out   = self.encoder(pixel_values=pixel_values)
        raw_features = vision_out.last_hidden_state

        emotions, emotion_probs = self.emotion_clf.predict(raw_features)
        emotion = emotions[0]

        image_features  = self.moe_proj(raw_features, lang_idx=lang_idx)
        image_features  = self.script_pos(image_features, lang_idx=lang_idx)

        if lang_name == "English":
            from transformers import BertTokenizer
            en_tok   = BertTokenizer.from_pretrained("bert-base-uncased")
            raw_caps = self.en_head.generate(image_features, en_tok, max_length)
        else:
            marian_features = self.moe_to_marian(image_features)
            lang_prefix     = MARIAN_LANG_PREFIX[lang_name]
            forced_bos      = tokenizer.convert_tokens_to_ids(lang_prefix)
            output_ids = self.marian.generate(
                inputs_embeds        = marian_features,
                forced_bos_token_id  = forced_bos,
                max_length           = max_length,
                num_beams            = 4,
                early_stopping       = True,
                no_repeat_ngram_size = 3,
            )
            raw_caps = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        decorated = [decorate_caption(c, emotion, lang_name) for c in raw_caps]
        return decorated, emotion, emotion_probs[0]
