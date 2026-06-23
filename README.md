# Multilingual Image Caption Generator

## Overview

Multilingual Image Caption Generator is a Deep Learning project that automatically generates image captions from uploaded images and translates them into multiple South Indian languages. The system also detects the emotion conveyed in the image and displays relevant emojis through an interactive Gradio web interface.

This project combines Computer Vision, Natural Language Processing, Machine Translation, and Vision-Language Models to create an intelligent multilingual image understanding system.

---

## Features

* Automatic image caption generation
* English caption generation using BLIP
* Translation into:

  * Tamil
  * Telugu
  * Kannada
  * Malayalam
* Emotion detection from images
* Emoji-based emotion representation
* Interactive Gradio web interface
* Zero-shot learning using pre-trained models
* No custom model training required

---

## System Architecture

```text
Input Image
     │
     ▼
BLIP Image Captioning
     │
     ▼
English Caption
     │
     ├───────────────┐
     │               │
     ▼               ▼
MarianMT       Emotion Detection
Translation    (Keyword Matching +
               CLIP Fallback)
     │               │
     ▼               ▼
Tamil           Emotion Label
Telugu                │
Kannada               ▼
Malayalam         Emoji Mapping
     │               │
     └───────┬───────┘
             ▼
        Gradio Web UI
```

---

## Models Used

### 1. BLIP (Bootstrapping Language-Image Pre-training)

**Model:** `Salesforce/blip-image-captioning-base`

**Purpose:**

* Generates English captions from images.

**Parameters:** ~247 Million

**Advantages:**

* State-of-the-art image captioning.
* Zero-shot capability.
* High-quality caption generation.

---

### 2. MarianMT (Marian Neural Machine Translation)

**Model:** `Helsinki-NLP/opus-mt-en-mul`

**Purpose:**

* Translates English captions into multiple South Indian languages.

**Supported Languages:**

* Tamil
* Telugu
* Kannada
* Malayalam

**Parameters:** ~76.9 Million

**Advantages:**

* Lightweight.
* Supports 100+ languages.
* Fast inference.

---

### 3. CLIP (Contrastive Language-Image Pretraining)

**Model:** `openai/clip-vit-base-patch32`

**Purpose:**

* Zero-shot emotion classification.
* Used when keyword matching fails.

**Parameters:** ~87.5 Million

**Advantages:**

* Understands both images and text.
* Strong vision-language reasoning capabilities.

---

## Technologies Used

* Python
* PyTorch
* Transformers
* Hugging Face
* BLIP
* MarianMT
* CLIP
* Gradio
* PIL
* NumPy

---

## Project Workflow

1. User uploads an image.
2. BLIP generates an English caption.
3. MarianMT translates the caption into:

   * Tamil
   * Telugu
   * Kannada
   * Malayalam
4. Emotion is detected using:

   * Keyword Matching
   * CLIP Fallback Classification
5. Appropriate emojis are assigned.
6. Results are displayed through Gradio.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/swamy16/Multilingual-Image-Caption-Generator.git
cd Multilingual-Image-Caption-Generator
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

---

## Sample Output

Input Image:

* Uploaded by user

Generated Results:

* English Caption
* Tamil Caption
* Telugu Caption
* Kannada Caption
* Malayalam Caption
* Emotion Label
* Emoji Representation

---

## Applications

* Assistive technology for visually impaired users
* Smart image understanding systems
* Social media content generation
* Educational tools
* Multilingual AI applications
* Image search and indexing

---

## Future Enhancements

* Hindi and other Indian language support
* Voice-based caption output
* Real-time webcam captioning
* Fine-tuned emotion classification
* Mobile application deployment
* Cloud deployment support

---

## Author

**Swamy,Shrekar,Tanish,Ullas**

B.Tech AIML Student At RV University 

---

## License

This project is developed for academic and educational purposes.
