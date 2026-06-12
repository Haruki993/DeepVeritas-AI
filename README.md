# 🔍 DeepVeritas AI

**Real vs AI-Generated Image Detection — powered by Vision Transformers**

DeepVeritas AI is a locally-deployable web application that detects whether an image is a genuine photograph or AI-generated/deepfake content. It uses a fine-tuned **ViT-Base/16** model and ships with four built-in Explainable AI (XAI) modules so you can *see* why the model made its decision — not just trust a black-box verdict.

---

## ✨ Features

- 🧠 **ViT-Base/16 classifier** fine-tuned on the [Parveshiiii/AI-vs-Real](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real) dataset
- 🟢🔴🟡 **Three-state verdict** — `REAL`, `FAKE`, or `UNCERTAIN` (confidence-gated at 70%)
- 🔬 **Explainable AI panel** with four independent analysis views:
  - **Attention Heatmap** — last-layer ViT CLS-token attention, overlaid on the image
  - **FFT Frequency Spectrum** — reveals generator upsampling artefacts
  - **RGB Pixel Distribution Histogram** — channel-wise tonal analysis
  - **Noise Residual Map** — Gaussian-subtraction camera-grain forensics
- 🖥️ **Fully local & private** — no cloud calls, no API keys, your images never leave your machine
- ⚡ **Streamlit UI** — drag, drop, done

---

## 📸 Preview

| Upload & Verdict | Deep Analysis |
|---|---|
| Real-time REAL / FAKE / UNCERTAIN classification with confidence scores | Attention maps, FFT spectrum, pixel histograms & noise residuals |

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Model | ViT-Base/16 (`google/vit-base-patch16-224-in21k`, fine-tuned) |
| Training | HuggingFace `Trainer`, FP16 mixed precision, on Kaggle (T4 GPU) |
| Inference | PyTorch + HuggingFace Transformers |
| Frontend | Streamlit |
| Analysis | NumPy, Matplotlib, SciPy, Pillow |
| Dataset | [Parveshiiii/AI-vs-Real](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real) (HuggingFace Hub) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- ~350MB free disk space for model weights

### Installation

```bash
git clone https://github.com/Haruki993/DeepVeritas-AI.git
cd DeepVeritas-AI

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> **Note:** The trained model files (`model.safetensors`, `config.json`, `preprocessor_config.json`) must be present in the `./models` directory. See [Model Setup](#-model-setup) below.

---

## 🧠 Model Setup

The fine-tuned model is too large for this repo and must be downloaded separately:

1. Download the model files from `[your model release / HF Hub link]`
2. Place them inside the `models/` directory:

```
models/
├── config.json
├── preprocessor_config.json
└── model.safetensors
```

---

## 📁 Project Structure

```
DeepVeritas-AI/
├── app.py                  # Streamlit application (UI + inference + XAI)
├── models/                 # Fine-tuned ViT model (not included — see Model Setup)
├── training/
│   └── notebook.ipynb      # Kaggle training notebook
├── requirements.txt
└── README.md
```

---

## 🔬 How It Works

1. User uploads an image (JPG / PNG / WEBP)
2. `ViTImageProcessor` resizes & normalizes to 224×224
3. `ViTForImageClassification` runs inference with `output_attentions=True`
4. Softmax → confidence scores → REAL / FAKE / UNCERTAIN verdict
5. Last-layer CLS attention is extracted, reshaped to a 14×14 grid, and overlaid as a heatmap
6. FFT, pixel histogram, and noise residual analyses run in parallel for forensic context

---

## 🎯 Model Performance

| Test Case | Verdict | Confidence |
|---|---|---|
| AI-generated portrait | FAKE | 98.7% |
| Genuine photograph | REAL | 97.6% |

---

## 🗺️ Roadmap

- [ ] Multi-class generator attribution (GAN / Diffusion / VAE)
- [ ] Video deepfake detection (frame-level + temporal consistency)
- [ ] FastAPI backend for integration
- [ ] Batch processing with PDF forensics reports
- [ ] Model quantization (INT8 / ONNX) for faster CPU inference

---

## 👥 Team

| Member | Role |
|---|---|
| Harshit Kumar Das | Team Lead — Full-stack development, model training & fine-tuning |
| Anuneet Dahiya | Research & Presentation |
| Manasvi Sharma | Dataset Curation |
| Sara Nema | Documentation |

---

## 📄 License

This project is developed for academic purposes as part of a final-year major project.

---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Parveshiiii/AI-vs-Real Dataset](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real)
- [Streamlit](https://streamlit.io/)
- Vision Transformer paper — Dosovitskiy et al., 2020
