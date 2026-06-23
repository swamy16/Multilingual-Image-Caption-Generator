import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from transformers import MarianMTModel, MarianTokenizer

# opus-mt-en-mul supports 100+ languages with >>lang<< prefixes
MODEL = "Helsinki-NLP/opus-mt-en-mul"
print(f"Loading {MODEL} ...")
tok   = MarianTokenizer.from_pretrained(MODEL)
model = MarianMTModel.from_pretrained(MODEL)
print(f"Vocab size: {tok.vocab_size}")

# South Indian language codes to try
candidates = {
    "Tamil":     [">>tam<<", ">>ta<<", ">>tam_Taml<<"],
    "Telugu":    [">>tel<<", ">>te<<", ">>tel_Telu<<"],
    "Kannada":   [">>kan<<", ">>kn<<", ">>kan_Knda<<"],
    "Malayalam": [">>mal<<", ">>ml<<", ">>mal_Mlym<<"],
}

print("\nChecking language prefix token IDs:")
found = {}
for lang, prefixes in candidates.items():
    for prefix in prefixes:
        tid = tok.convert_tokens_to_ids(prefix)
        if tid != tok.unk_token_id and tid != 3:
            print(f"  {lang:10s}  {prefix}  -> id={tid}  ✓")
            found[lang] = prefix
            break
    if lang not in found:
        print(f"  {lang:10s}  NOT FOUND in vocab")

print("\nTest translations:")
import torch
for lang, prefix in found.items():
    src = [f"{prefix} A beautiful temple in South India"]
    inputs = tok(src, return_tensors="pt", padding=True)
    with torch.no_grad():
        ids = model.generate(**inputs, max_length=60, num_beams=4)
    result = tok.batch_decode(ids, skip_special_tokens=True)
    print(f"  {lang:10s}: {result[0]}")
