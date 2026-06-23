import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datasets import load_dataset

sources = [
    ("jxie/flickr8k",                    "test",  {}),
    ("jxie/flickr8k",                    "train", {}),
    ("clip-benchmark/wds_flickr8k",      "test",  {}),
    ("atasoglu/flickr8k-dataset",        "train", {}),
    ("Multimodal-Fatima/Flickr30k_train","train", {}),
    ("nlphuji/flickr30k",                "test",  {}),
]

for path, split, kwargs in sources:
    try:
        ds = load_dataset(path, split=split, trust_remote_code=True, **kwargs)
        print(f"✓ {path} [{split}]  len={len(ds)}  cols={ds.column_names}")
        sample = ds[0]
        for k, v in sample.items():
            print(f"    {k}: {type(v).__name__} = {str(v)[:80]}")
        break
    except Exception as e:
        print(f"✗ {path}: {str(e)[:80]}")
