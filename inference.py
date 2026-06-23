import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
from PIL import Image
from pipeline import SouthIndianCaptioner
from emotion import EMOTIONS

def main():
    parser = argparse.ArgumentParser(description="South Indian Image Captioner")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--lang", default="all",
                        help="Language: Tamil/Telugu/Kannada/Malayalam/English/all")
    args = parser.parse_args()

    captioner = SouthIndianCaptioner()
    image     = Image.open(args.image).convert("RGB")
    result    = captioner.caption(image)

    print("\n" + "=" * 60)
    print(f"  Image   : {args.image}")
    print(f"  BLIP    : {result['english_raw']}")
    print(f"  Emotion : {result['emotion']}")
    print("\n  Confidence:")
    for emo, prob in zip(EMOTIONS, result["probs"]):
        bar = "█" * int(prob.item() * 30)
        print(f"    {emo:12s} {bar:<30s} {prob.item()*100:.1f}%")
    print("\n  Captions:")
    print("  " + "-" * 56)

    langs = (list(result["captions"].keys())
             if args.lang == "all"
             else [args.lang])

    for lang in langs:
        if lang in result["captions"]:
            print(f"  {lang:10s} │ {result['captions'][lang]}")

    print("=" * 60)

if __name__ == "__main__":
    main()
