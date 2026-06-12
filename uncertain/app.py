import streamlit as st
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepVeritas AI",
    page_icon="🔍",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d0d0f;
        color: #e2e2e8;
    }
    .stApp { background-color: #0d0d0f; }

    .header {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
    }
    .header h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        color: #e879f9;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
    }
    .header p { color: #888; font-size: 0.95rem; margin: 0; }

    [data-testid="stFileUploadDropzone"] {
        background-color: #141418 !important;
        border: 1.5px dashed #333 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #e879f9 !important;
    }

    .result-card {
        border-radius: 14px;
        padding: 1.6rem 2rem 1.2rem 2rem;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .result-real {
        background: linear-gradient(135deg, #0f2a1f 0%, #0a1a14 100%);
        border: 1.5px solid #22c55e;
    }
    .result-fake {
        background: linear-gradient(135deg, #2a0f0f 0%, #1a0a0a 100%);
        border: 1.5px solid #ef4444;
    }
    .result-uncertain {
        background: linear-gradient(135deg, #2a220f 0%, #1a150a 100%);
        border: 1.5px solid #eab308;
    }
    .result-label {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-real .result-label { color: #22c55e; }
    .result-fake .result-label { color: #ef4444; }
    .result-uncertain .result-label { color: #eab308; }
    .result-sub { font-size: 0.88rem; color: #888; margin: 0; }

    .conf-label { font-size: 0.82rem; color: #aaa; margin-bottom: 0.2rem; }
    .conf-pct-real { color: #22c55e; font-weight: 600; }
    .conf-pct-fake { color: #ef4444; font-weight: 600; }

    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #e879f9;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 2rem 0 0.8rem 0;
    }
    .analysis-caption {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.4rem;
        line-height: 1.5;
    }

    .divider { border: none; border-top: 1px solid #1e1e24; margin: 1.8rem 0; }
    .footer { text-align: center; color: #444; font-size: 0.78rem; padding: 2rem 0 1rem; }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#141418",
    "axes.facecolor":    "#141418",
    "axes.edgecolor":    "#333",
    "axes.labelcolor":   "#aaa",
    "xtick.color":       "#666",
    "ytick.color":       "#666",
    "text.color":        "#e2e2e8",
    "grid.color":        "#222",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def fig_to_pil(fig):
    """Convert a matplotlib figure to a PIL image for st.image()."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).copy()

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    MODEL_PATH = "./models"
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    model = ViTForImageClassification.from_pretrained(MODEL_PATH, attn_implementation="eager")
    model.eval()
    return model, processor

# ── Inference + attention ─────────────────────────────────────────────────────
def predict(image: Image.Image, model, processor):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    probs     = torch.softmax(outputs.logits, dim=1)[0]
    pred_idx  = torch.argmax(probs).item()
    label     = model.config.id2label[pred_idx]
    is_real   = label.lower() in ("realism", "real")
    conf_real = probs[0].item() if model.config.id2label[0].lower() in ("realism", "real") else probs[1].item()
    conf_fake = 1.0 - conf_real

    # ── Attention map with fallback ───────────────────────────────────────────
    attn_map = None

    if outputs.attentions and len(outputs.attentions) > 0:
        # Normal path — ViT returned attention weights
        last_attn = outputs.attentions[-1]           # (1, heads, 197, 197)
        cls_attn  = last_attn[0].mean(0)[0, 1:]      # mean over heads → (196,)
        grid_size = int(cls_attn.shape[0] ** 0.5)
        attn_map  = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
    else:
        # Fallback — gradient saliency (works on any model)
        img_tensor = inputs["pixel_values"].requires_grad_(True)
        logits     = model(pixel_values=img_tensor).logits
        logits[0, pred_idx].backward()
        saliency   = img_tensor.grad[0].abs().mean(0)  # mean over channels → (H, W)
        attn_map   = saliency.cpu().numpy()

    # Normalise to [0, 1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    return is_real, conf_real, conf_fake, attn_map

# ── Analysis plots ────────────────────────────────────────────────────────────
def plot_attention(image: Image.Image, attn_map: np.ndarray) -> Image.Image:
    """Overlay attention heatmap on the original image."""
    img_rgb  = np.array(image.convert("RGB").resize((224, 224)))
    heat     = Image.fromarray(attn_map).resize((224, 224), Image.BILINEAR)
    heat_arr = np.array(heat)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(img_rgb);           axes[0].set_title("Original",  fontsize=9); axes[0].axis("off")
    axes[1].imshow(heat_arr, cmap="magma"); axes[1].set_title("Attention", fontsize=9); axes[1].axis("off")

    # Overlay
    colored = cm.magma(heat_arr)[:, :, :3]
    overlay = (0.55 * img_rgb / 255.0 + 0.45 * colored)
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay);           axes[2].set_title("Overlay",   fontsize=9); axes[2].axis("off")

    fig.tight_layout(pad=0.5)
    result = fig_to_pil(fig)
    plt.close(fig)
    return result


def plot_fft(image: Image.Image) -> Image.Image:
    """2-D FFT frequency spectrum — AI images show grid artefacts."""
    gray     = np.array(image.convert("L").resize((224, 224))).astype(np.float32)
    fft      = np.fft.fftshift(np.fft.fft2(gray))
    spectrum = np.log1p(np.abs(fft))

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    im0 = axes[0].imshow(spectrum, cmap="inferno")
    axes[0].set_title("FFT Magnitude Spectrum", fontsize=9)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Radial power profile
    cy, cx  = np.array(spectrum.shape) // 2
    Y, X    = np.ogrid[:spectrum.shape[0], :spectrum.shape[1]]
    R       = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    radial  = np.bincount(R.ravel(), weights=spectrum.ravel()) / (np.bincount(R.ravel()) + 1e-8)
    freqs   = np.arange(len(radial))

    axes[1].plot(freqs[:80], radial[:80], color="#e879f9", linewidth=1.5)
    axes[1].set_title("Radial Power Profile", fontsize=9)
    axes[1].set_xlabel("Spatial frequency")
    axes[1].set_ylabel("Mean log power")
    axes[1].grid(True)

    fig.tight_layout(pad=0.5)
    result = fig_to_pil(fig)
    plt.close(fig)
    return result


def plot_histogram(image: Image.Image) -> Image.Image:
    """Per-channel pixel histogram. AI images tend to have unnaturally smooth distributions."""
    img_arr = np.array(image.convert("RGB").resize((224, 224)))
    colors  = [("#ef4444", "Red"), ("#22c55e", "Green"), ("#60a5fa", "Blue")]

    fig, ax = plt.subplots(figsize=(7, 3))
    for i, (color, label) in enumerate(colors):
        hist, bins = np.histogram(img_arr[:, :, i].ravel(), bins=64, range=(0, 255))
        centers    = (bins[:-1] + bins[1:]) / 2
        ax.plot(centers, hist, color=color, label=label, linewidth=1.5, alpha=0.85)

    ax.set_title("RGB Pixel Distribution", fontsize=9)
    ax.set_xlabel("Pixel value (0–255)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, framealpha=0.2)
    ax.grid(True)

    fig.tight_layout(pad=0.5)
    result = fig_to_pil(fig)
    plt.close(fig)
    return result


def plot_noise(image: Image.Image) -> Image.Image:
    """High-frequency noise residual — AI images show structured/repeating noise."""
    gray   = np.array(image.convert("L").resize((224, 224))).astype(np.float32)
    # Simple Laplacian-style noise extraction
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(gray, sigma=2)
    noise  = gray - smooth

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    im0 = axes[0].imshow(noise, cmap="RdBu_r", vmin=-30, vmax=30)
    axes[0].set_title("Noise Residual", fontsize=9)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].hist(noise.ravel(), bins=80, color="#e879f9", alpha=0.8, edgecolor="none")
    axes[1].set_title("Noise Distribution", fontsize=9)
    axes[1].set_xlabel("Residual value")
    axes[1].set_ylabel("Count")
    axes[1].grid(True)

    fig.tight_layout(pad=0.5)
    result = fig_to_pil(fig)
    plt.close(fig)
    return result

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🔍 DeepVeritas AI</h1>
    <p>Upload an image to detect if it's real or AI-generated</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading model..."):
    model, processor = load_model()

uploaded = st.file_uploader(
    "Drop an image here, or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, width="stretch")

    with st.spinner("Analyzing..."):
        is_real, conf_real, conf_fake, attn_map = predict(image, model, processor)

    max_conf = max(conf_real, conf_fake)
    if max_conf < 0.70:
        verdict    = "UNCERTAIN"
        card_class = "result-uncertain"
        icon       = "🤔"
        subtext    = "The model is uncertain. This could be a heavily edited real photo or a highly realistic deepfake."
    else:
        verdict    = "REAL"        if is_real else "FAKE"
        card_class = "result-real" if is_real else "result-fake"
        icon       = "✅"           if is_real else "⚠️"
        subtext    = "This image appears to be a genuine photograph." if is_real \
                     else "This image appears to be AI-generated or manipulated."

    # ── Verdict card ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-label">{icon} {verdict}</div>
        <div class="result-sub">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence bars ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_r, col_f = st.columns(2)
    with col_r:
        st.markdown(f'<div class="conf-label">Real &nbsp;<span class="conf-pct-real">{conf_real*100:.1f}%</span></div>', unsafe_allow_html=True)
        st.progress(float(conf_real))
    with col_f:
        st.markdown(f'<div class="conf-label">Fake &nbsp;<span class="conf-pct-fake">{conf_fake*100:.1f}%</span></div>', unsafe_allow_html=True)
        st.progress(float(conf_fake))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Deep Analysis Section ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">🧠 Deep Analysis</div>', unsafe_allow_html=True)

    with st.spinner("Generating analysis plots..."):
        attn_img  = plot_attention(image, attn_map)
        fft_img   = plot_fft(image)
        hist_img  = plot_histogram(image)
        noise_img = plot_noise(image)

    # 1. Attention map
    st.image(attn_img, width="stretch")
    st.markdown("""
    <div class="analysis-caption">
    <b>Attention Heatmap</b> — Shows which regions of the image the model focused on most.
    Bright areas = high attention. AI-generated images often show uniform or unnatural attention patterns
    (e.g. focused on textures or backgrounds instead of meaningful features).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. FFT spectrum
    st.image(fft_img, width="stretch")
    st.markdown("""
    <div class="analysis-caption">
    <b>FFT Frequency Spectrum</b> — Converts the image into frequency space.
    AI-generated images often show grid-like artefacts or unnatural spikes at regular intervals
    caused by upsampling in the generator network. The radial power profile shows how energy
    is distributed across spatial frequencies.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. RGB histogram
    st.image(hist_img, width="stretch")
    st.markdown("""
    <div class="analysis-caption">
    <b>RGB Pixel Distribution</b> — Shows how pixel values are spread across each colour channel.
    Real photographs tend to have irregular, camera-noise-influenced distributions.
    AI-generated images often produce unnaturally smooth or symmetrical histograms.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Noise residual
    st.image(noise_img, width="stretch")
    st.markdown("""
    <div class="analysis-caption">
    <b>Noise Residual</b> — Extracts high-frequency noise by subtracting a smoothed version of the image.
    Real cameras produce random grain. AI generators often leave behind structured, repeating, or
    suspiciously regular noise patterns that are invisible to the eye but visible here.
    </div>
    """, unsafe_allow_html=True)

    # ── Image details expander ────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    with st.expander("Image details"):
        st.write(f"**Filename:** {uploaded.name}")
        st.write(f"**Dimensions:** {image.width} × {image.height} px")
        st.write(f"**Format:** {image.format or uploaded.type}")
        st.write(f"**Real confidence:** {conf_real*100:.2f}%")
        st.write(f"**Fake confidence:** {conf_fake*100:.2f}%")

st.markdown('<div class="footer">DeepVeritas AI v2 &middot; High-Res ViT &middot; Multi-Generator Dataset</div>', unsafe_allow_html=True)