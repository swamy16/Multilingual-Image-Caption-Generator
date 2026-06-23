import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import gradio as gr
from PIL import Image
from pipeline import SouthIndianCaptioner
from emotion import EMOTIONS

EMOTION_COLORS = {
    "Happy":      "#FFD700",
    "Sad":        "#6495ED",
    "Peaceful":   "#90EE90",
    "Exciting":   "#FF6347",
    "Fearful":    "#9370DB",
    "Surprising": "#FF69B4",
}

EMOTION_BANNERS = {
    "Happy":      "😊 Happy",
    "Sad":        "😢 Sad",
    "Peaceful":   "😌 Peaceful",
    "Exciting":   "🤩 Exciting",
    "Fearful":    "😨 Fearful",
    "Surprising": "😮 Surprising",
}

print("Initialising pipeline...")
captioner = SouthIndianCaptioner()
print("Ready!")


def build_emotion_bar(probs) -> str:
    html = "<div style='font-family:monospace;font-size:14px;line-height:2'>"
    for emo, prob in zip(EMOTIONS, probs):
        pct   = prob.item() * 100
        width = int(pct * 2)
        color = EMOTION_COLORS.get(emo, "#ccc")
        html += (
            f"<div>"
            f"<span style='display:inline-block;width:110px'>{emo}</span>"
            f"<span style='display:inline-block;background:{color};"
            f"width:{width}px;height:18px;border-radius:4px;vertical-align:middle'></span>"
            f"&nbsp;<b>{pct:.1f}%</b></div>"
        )
    html += "</div>"
    return html


def run_all(image):
    if image is None:
        return "Please upload an image.", "", ""
    result  = captioner.caption(image)
    md      = f"**English (BLIP):** {result['english_raw']}\n\n---\n\n"
    for lang, cap in result["captions"].items():
        md += f"**{lang}**\n\n{cap}\n\n"
    emotion = EMOTION_BANNERS.get(result["emotion"], result["emotion"])
    bar     = build_emotion_bar(result["probs"])
    return md, emotion, bar


def run_single(image, language):
    if image is None:
        return "Please upload an image.", "", ""
    result  = captioner.caption(image)
    caption = result["captions"][language]
    emotion = EMOTION_BANNERS.get(result["emotion"], result["emotion"])
    bar     = build_emotion_bar(result["probs"])
    return caption, emotion, bar


# ── UI ────────────────────────────────────────────────────────
with gr.Blocks(title="South Indian Caption Generator") as demo:

    gr.Markdown(
        "# South Indian Multilingual Image Caption Generator\n"
        "Powered by **BLIP** + **MarianMT** + **Emotion Detection**\n\n"
        "Supports: Tamil · Telugu · Kannada · Malayalam · English"
    )

    with gr.Tabs():

        with gr.Tab("All Languages"):
            with gr.Row():
                with gr.Column():
                    img_all = gr.Image(type="pil", label="Upload Image")
                    btn_all = gr.Button("Generate All Captions", variant="primary")
                with gr.Column():
                    out_md  = gr.Markdown()
                    out_emo = gr.Textbox(label="Detected Emotion")
                    out_bar = gr.HTML()
            btn_all.click(
                fn=run_all,
                inputs=[img_all],
                outputs=[out_md, out_emo, out_bar]
            )

        with gr.Tab("Single Language"):
            with gr.Row():
                with gr.Column():
                    img_s  = gr.Image(type="pil", label="Upload Image")
                    lang_s = gr.Dropdown(
                        choices=["Tamil", "Telugu", "Kannada", "Malayalam", "English"],
                        value="Tamil",
                        label="Language"
                    )
                    btn_s  = gr.Button("Generate Caption", variant="primary")
                with gr.Column():
                    out_cap = gr.Textbox(label="Caption", lines=3)
                    out_emo2 = gr.Textbox(label="Detected Emotion")
                    out_bar2 = gr.HTML()
            btn_s.click(
                fn=run_single,
                inputs=[img_s, lang_s],
                outputs=[out_cap, out_emo2, out_bar2]
            )

    gr.Markdown(
        "**Emotions:** 😊 Happy · 😢 Sad · 😌 Peaceful · 🤩 Exciting · 😨 Fearful · 😮 Surprising"
    )

if __name__ == "__main__":
    demo.launch(share=True)
