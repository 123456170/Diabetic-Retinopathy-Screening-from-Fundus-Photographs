"""
Diabetic Retinopathy Screening from Fundus Photographs
-----------------------------------------------------
Streamlit demo / simulation application.

Important:
- This is an educational and prototyping demo only.
- It does not use real patient data.
- It is not a medical device and must not be used for clinical decisions.
- All images, scores, probabilities, and recommendations are synthetic.
- A qualified clinician must remain responsible for any real screening decision.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


DEFAULTS: Dict[str, Any] = {
    "width": _env_int("DRS_WIDTH", 384),
    "frames": _env_int("DRS_FRAMES", 42),
    "fps": _env_int("DRS_FPS", 12),
    "seed": _env_int("DRS_SEED", 7),
    "severity": _env_float("DRS_SEVERITY", 0.62),
    "quality": _env_float("DRS_QUALITY", 0.78),
    "noise": _env_float("DRS_NOISE", 0.18),
    "blur": _env_float("DRS_BLUR", 0.12),
    "exposure": _env_float("DRS_EXPOSURE", 0.62),
    "simulate_progression": _env_bool("DRS_SIMULATE_PROGRESSION", True),
    "refer_threshold": _env_float("DRS_REFER_THRESHOLD", 0.65),
    "review_threshold": _env_float("DRS_REVIEW_THRESHOLD", 0.35),
    "min_quality": _env_float("DRS_MIN_QUALITY", 0.45),
}


# -----------------------------------------------------------------------------
# Typed parameter container
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoParams:
    width: int
    height: int
    frames: int
    fps: int
    seed: int
    eye: str
    severity: float
    quality: float
    noise: float
    blur: float
    exposure: float
    simulate_progression: bool
    refer_threshold: float
    review_threshold: float
    min_quality: float


# -----------------------------------------------------------------------------
# Safe image helpers
# -----------------------------------------------------------------------------


def _pil_gaussian_blur(rgb: np.ndarray, radius: float) -> np.ndarray:
    """
    Apply Gaussian blur using Pillow instead of OpenCV.

    This avoids OpenCV optimized filter paths that can raise unsupported
    source/destination format errors on some builds.
    """
    if radius <= 0.01:
        return rgb.copy()

    img = Image.fromarray(rgb, "RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=float(max(0.1, radius))))
    return np.array(img, dtype=np.uint8)


def _numpy_laplacian(gray: np.ndarray) -> np.ndarray:
    """
    Compute a simple 4-neighbor Laplacian using NumPy.

    This replaces cv2.Laplacian and avoids OpenCV filter-format issues.
    """
    padded = np.pad(gray.astype(np.float64), ((1, 1), (1, 1)), mode="edge")

    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )

    return laplacian


# -----------------------------------------------------------------------------
# Synthetic fundus rendering
# -----------------------------------------------------------------------------


def _draw_vessel(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    angle_deg: float,
    length: float,
    width: float,
    rng: np.random.Generator,
    depth: int,
) -> None:
    """
    Recursively draw a simplified retinal vessel tree.

    This is intentionally simple and deterministic for a given RNG state.
    It is for visual simulation only, not anatomical accuracy.
    """
    if depth <= 0 or length < 3.0 or width < 0.8:
        return

    rad = math.radians(angle_deg)
    x2 = x + math.cos(rad) * length
    y2 = y + math.sin(rad) * length

    draw.line(
        [(int(x), int(y)), (int(x2), int(y2))],
        fill=(104, 24, 24),
        width=max(1, int(width)),
    )

    if depth > 1:
        branch_count = int(rng.integers(1, 3))
        for _ in range(branch_count):
            new_angle = angle_deg + float(rng.normal(0.0, 22.0))
            new_length = length * float(rng.uniform(0.55, 0.78))
            new_width = width * 0.66
            _draw_vessel(
                draw=draw,
                x=x2,
                y=y2,
                angle_deg=new_angle,
                length=new_length,
                width=new_width,
                rng=rng,
                depth=depth - 1,
            )


def render_fundus(
    params: DemoParams,
    severity: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Render a synthetic fundus-like image.

    Returns:
        RGB image as uint8 numpy array and a list of synthetic lesion markers.
    """
    w, h = params.width, params.height
    short = min(w, h)

    texture = rng.normal(0.0, 1.2, size=(h, w)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    disc_x = w * (0.28 if params.eye == "Left" else 0.72)
    disc_y = h * 0.50
    macula_x = w * (0.68 if params.eye == "Left" else 0.32)
    macula_y = h * 0.52

    cx = (disc_x + macula_x) / 2.0
    cy = (disc_y + macula_y) / 2.0
    max_r = math.hypot(w, h) * 0.58

    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_r
    r = np.clip(r, 0.0, 1.0)

    red = 152.0 - 68.0 * r + texture
    green = 78.0 - 34.0 * r + 0.65 * texture
    blue = 58.0 - 24.0 * r + 0.45 * texture

    arr = np.stack([red, green, blue], axis=-1)
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    pil = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(pil)

    disc_r = int(short * 0.085)

    for scale, color in [
        (2.25, (165, 95, 75)),
        (1.75, (190, 125, 95)),
        (1.28, (215, 165, 125)),
    ]:
        rr = int(disc_r * scale)
        draw.ellipse(
            [
                int(disc_x - rr),
                int(disc_y - rr),
                int(disc_x + rr),
                int(disc_y + rr),
            ],
            fill=color,
        )

    draw.ellipse(
        [
            int(disc_x - disc_r),
            int(disc_y - disc_r),
            int(disc_x + disc_r),
            int(disc_y + disc_r),
        ],
        fill=(238, 218, 176),
        outline=(170, 130, 95),
        width=max(1, int(short * 0.004)),
    )

    cup_r = int(disc_r * 0.45)
    draw.ellipse(
        [
            int(disc_x - cup_r),
            int(disc_y - cup_r),
            int(disc_x + cup_r),
            int(disc_y + cup_r),
        ],
        fill=(248, 238, 218),
    )

    macula_r = int(short * 0.075)
    draw.ellipse(
        [
            int(macula_x - macula_r),
            int(macula_y - macula_r),
            int(macula_x + macula_r),
            int(macula_y + macula_r),
        ],
        fill=(92, 44, 44),
    )

    fovea_r = int(macula_r * 0.22)
    draw.ellipse(
        [
            int(macula_x - fovea_r),
            int(macula_y - fovea_r),
            int(macula_x + fovea_r),
            int(macula_y + fovea_r),
        ],
        fill=(125, 72, 72),
    )

    vessel_width = max(1, int(short * 0.016))
    for angle in (-62, -34, -8, 18, 46, 70):
        _draw_vessel(
            draw=draw,
            x=disc_x,
            y=disc_y,
            angle_deg=float(angle),
            length=short * 0.30,
            width=float(vessel_width),
            rng=rng,
            depth=5,
        )

    lesions: List[Dict[str, Any]] = []
    count = int(math.ceil(_clip01(severity) * short * 0.12))

    for _ in range(count):
        x = float(rng.uniform(w * 0.15, w * 0.85))
        y = float(rng.uniform(h * 0.15, h * 0.85))

        if math.hypot(x - disc_x, y - disc_y) < disc_r * 1.25:
            continue

        u = float(rng.random())
        score = float(_clip01(severity * rng.uniform(0.45, 1.0)))

        if u < 0.56:
            radius = max(1, int(short * 0.006 * (0.7 + score)))
            color = (190, 28, 38)
            lesion_type = "microaneurysm"
        elif u < 0.82:
            radius = max(2, int(short * float(rng.uniform(0.010, 0.026)) * (0.6 + score)))
            color = (138, 12, 24)
            lesion_type = "hemorrhage"
        else:
            radius = max(2, int(short * float(rng.uniform(0.008, 0.020)) * (0.6 + score)))
            color = (236, 222, 154)
            lesion_type = "exudate"

        draw.ellipse(
            [
                int(x - radius),
                int(y - radius),
                int(x + radius),
                int(y + radius),
            ],
            fill=color,
        )

        lesions.append(
            {
                "x": x,
                "y": y,
                "score": score,
                "type": lesion_type,
            }
        )

    return np.array(pil, dtype=np.uint8), lesions


# -----------------------------------------------------------------------------
# Degradation, quality assessment, pseudo-model, heatmap, overlay
# -----------------------------------------------------------------------------


def apply_degradations(
    rgb: np.ndarray,
    params: DemoParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply simulated blur, noise, and exposure changes.
    """
    img = rgb.astype(np.float32)

    brightness = 0.55 + 0.95 * _clip01(params.exposure)
    img *= brightness

    eff_blur = _clip01(params.blur + (1.0 - _clip01(params.quality)) * 0.65)
    eff_noise = _clip01(params.noise + (1.0 - _clip01(params.quality)) * 0.22)

    if eff_blur > 0.02:
        short = min(params.width, params.height)
        radius = eff_blur * short / 55.0
        img_uint8 = np.clip(img, 0.0, 255.0).astype(np.uint8)
        img = _pil_gaussian_blur(img_uint8, radius).astype(np.float32)

    if eff_noise > 0.02:
        noise = rng.normal(0.0, eff_noise * 18.0, size=img.shape).astype(np.float32)
        img += noise

    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def assess_quality(
    rgb: np.ndarray,
    params: DemoParams,
) -> Dict[str, float]:
    """
    Compute simple image-quality proxies.

    These are not validated clinical quality metrics. They are deterministic
    simulation signals for demonstrating a quality gate.
    """
    gray = np.asarray(
        Image.fromarray(rgb, "RGB").convert("L"),
        dtype=np.uint8,
    )

    gray_f = gray.astype(np.float64)
    lap = _numpy_laplacian(gray_f)

    lap_var = float(np.var(lap))
    expected = 260.0 * (float(params.width) / 384.0) ** 2

    sharpness = _clip01(lap_var / max(1.0, expected))

    brightness = float(gray_f.mean())
    brightness_score = _clip01(1.0 - abs(brightness - 115.0) / 95.0)

    contrast = float(gray_f.std())
    contrast_score = _clip01(contrast / 52.0)

    noise_metric = float(np.median(np.abs(lap)))
    noise_score = _clip01(1.0 - noise_metric / max(1.0, expected * 0.08))

    quality_score = _clip01(
        0.45 * sharpness
        + 0.20 * brightness_score
        + 0.20 * contrast_score
        + 0.15 * noise_score
    )

    return {
        "quality_score": quality_score,
        "sharpness": sharpness,
        "brightness": brightness,
        "brightness_score": brightness_score,
        "contrast": contrast,
        "contrast_score": contrast_score,
        "noise_metric": noise_metric,
        "noise_score": noise_score,
    }


def predict_referability(
    severity: float,
    quality_score: float,
    params: DemoParams,
) -> Tuple[float, float]:
    """
    Produce a synthetic referable probability and uncertainty score.

    This is not a trained model. It is a transparent simulation rule used to
    demonstrate downstream decision support behavior.
    """
    sev = _clip01(severity)
    q = _clip01(quality_score)

    logit = -1.9 + 5.4 * sev - 2.3 * (1.0 - q)
    logit = max(-30.0, min(30.0, logit))

    probability = float(1.0 / (1.0 + math.exp(-logit)))

    uncertainty = float(
        _clip01(
            0.55 * (1.0 - q)
            + 0.45 * abs(probability - 0.5)
            + 0.15 * _clip01(params.blur)
        )
    )

    return probability, uncertainty


def decide_action(
    probability: float,
    quality_score: float,
    uncertainty: float,
    params: DemoParams,
) -> str:
    """
    Map synthetic scores to a screening action.

    Actions:
    - Repeat image: quality is too poor for reliable interpretation.
    - Refer to ophthalmology: high synthetic referable probability.
    - Clinician review: borderline probability or high uncertainty.
    - Routine follow-up: low synthetic probability and adequate quality.
    """
    refer_t = _clip01(params.refer_threshold)
    review_t = min(_clip01(params.review_threshold), refer_t)
    min_q = _clip01(params.min_quality)

    if quality_score < min_q:
        return "Repeat image"

    if uncertainty > 0.75 and probability < refer_t:
        return "Clinician review"

    if probability >= refer_t:
        return "Refer to ophthalmology"

    if probability >= review_t:
        return "Clinician review"

    return "Routine follow-up"


def make_heatmap(
    rgb: np.ndarray,
    lesions: List[Dict[str, Any]],
    params: DemoParams,
) -> np.ndarray:
    """
    Create a synthetic explanation heatmap from lesion markers.

    This is not a Grad-CAM or model-derived attribution map. It is a simple
    visual aid for the simulation.
    """
    h, w = rgb.shape[:2]
    short = min(h, w)

    mask_img = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(mask_img)

    for lesion in lesions:
        score = _clip01(float(lesion["score"]))
        radius = max(2, int(short * 0.018 * score))
        intensity = max(1, int(255 * score))

        x = float(lesion["x"])
        y = float(lesion["y"])

        mask_draw.ellipse(
            [
                int(x - radius),
                int(y - radius),
                int(x + radius),
                int(y + radius),
            ],
            fill=intensity,
        )

    extrema = mask_img.getextrema()
    if extrema[1] <= 0:
        radius = int(short * 0.16)
        mask_draw.ellipse(
            [
                int(w / 2 - radius),
                int(h / 2 - radius),
                int(w / 2 + radius),
                int(h / 2 + radius),
            ],
            fill=45,
        )

    blur_radius = max(1.0, short * 0.022)
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    mask = np.asarray(mask_img, dtype=np.float32)
    mask_max = float(mask.max())

    if mask_max <= 0.0:
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
    else:
        mask_u8 = np.clip(mask * (255.0 / mask_max), 0.0, 255.0).astype(np.uint8)

    heat_bgr = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def add_overlay(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    frame_index: int,
    severity: float,
    quality: Dict[str, float],
    probability: float,
    uncertainty: float,
    action: str,
    params: DemoParams,
) -> np.ndarray:
    """
    Add a synthetic HUD overlay with decision-support style information.
    """
    base = Image.fromarray(rgb, "RGB")
    heat = Image.fromarray(heatmap, "RGB")
    blended = Image.blend(base, heat, alpha=0.22).convert("RGBA")

    panel = Image.new("RGBA", blended.size, (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(panel)

    panel_w = int(params.width * 0.50)
    panel_h = int(params.height * 0.40)
    pdraw.rectangle([8, 8, 8 + panel_w, 8 + panel_h], fill=(0, 0, 0, 155))

    blended = Image.alpha_composite(blended, panel)
    draw = ImageDraw.Draw(blended)

    action_colors = {
        "Refer to ophthalmology": (220, 60, 60, 255),
        "Clinician review": (245, 180, 50, 255),
        "Routine follow-up": (60, 200, 120, 255),
        "Repeat image": (80, 160, 255, 255),
    }
    border_color = action_colors.get(action, (220, 220, 220, 255))

    border_width = max(2, int(min(params.width, params.height) * 0.012))
    draw.rectangle(
        [2, 2, params.width - 3, params.height - 3],
        outline=border_color,
        width=border_width,
    )

    lines = [
        f"Frame {frame_index + 1}/{params.frames}",
        f"Eye: {params.eye}",
        f"Quality: {quality['quality_score']:.2f}",
        f"Referable prob: {probability:.2f}",
        f"Uncertainty: {uncertainty:.2f}",
        f"Action: {action}",
    ]

    y = 16
    line_height = max(13, int(params.height * 0.058))
    for line in lines:
        draw.text((18, y), line, fill=(255, 255, 255, 255))
        y += line_height

    draw.text(
        (10, params.height - 20),
        "SIMULATION ONLY - NOT FOR CLINICAL USE",
        fill=(255, 255, 255, 230),
    )

    return np.array(blended.convert("RGB"), dtype=np.uint8)


def compose_frame(
    params: DemoParams,
    frame_index: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compose one synthetic demo frame and its structured metadata row.
    """
    total_frames = max(1, params.frames)
    t = frame_index / float(total_frames - 1) if total_frames > 1 else 0.0

    severity = params.severity
    if params.simulate_progression:
        severity = float(_clip01(params.severity * (0.35 + 0.95 * t)))

    rng = np.random.default_rng(params.seed + frame_index)

    base_rgb, lesions = render_fundus(params, severity, rng)
    degraded_rgb = apply_degradations(base_rgb, params, rng)

    quality = assess_quality(degraded_rgb, params)
    probability, uncertainty = predict_referability(
        severity=severity,
        quality_score=quality["quality_score"],
        params=params,
    )
    action = decide_action(
        probability=probability,
        quality_score=quality["quality_score"],
        uncertainty=uncertainty,
        params=params,
    )

    heatmap = make_heatmap(degraded_rgb, lesions, params)
    overlay = add_overlay(
        rgb=degraded_rgb,
        heatmap=heatmap,
        frame_index=frame_index,
        severity=severity,
        quality=quality,
        probability=probability,
        uncertainty=uncertainty,
        action=action,
        params=params,
    )

    row: Dict[str, Any] = {
        "frame_index": int(frame_index),
        "time_seconds": round(float(frame_index / max(1, params.fps)), 3),
        "eye": params.eye,
        "severity_effective": round(float(severity), 4),
        "quality_score": round(float(quality["quality_score"]), 4),
        "sharpness": round(float(quality["sharpness"]), 4),
        "brightness_score": round(float(quality["brightness_score"]), 4),
        "contrast_score": round(float(quality["contrast_score"]), 4),
        "noise_score": round(float(quality["noise_score"]), 4),
        "referable_probability": round(float(probability), 4),
        "uncertainty": round(float(uncertainty), 4),
        "action": action,
        "lesion_count": int(len(lesions)),
    }

    return overlay, row


# -----------------------------------------------------------------------------
# Video generation
# -----------------------------------------------------------------------------


def _open_video_writer(
    video_dir: str,
    fps: int,
    size: Tuple[int, int],
) -> Tuple[Any, str]:
    """
    Try several MP4-compatible OpenCV codecs and return the first working writer.
    """
    errors: List[str] = []

    for codec in ("avc1", "H264", "mp4v"):
        path = os.path.join(video_dir, f"demo_{codec}.mp4")
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(path, fourcc, float(fps), size)
        except Exception as exc:
            errors.append(f"{codec}: {exc}")
            continue

        if writer.isOpened():
            return writer, path

        try:
            writer.release()
        except Exception:
            pass

        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

        errors.append(f"{codec}: writer did not open")

    raise RuntimeError(
        "OpenCV could not open an MP4 video writer. "
        "Tried codecs: " + ", ".join(errors)
    )


def generate_demo_video(params: DemoParams) -> Tuple[str, str, pd.DataFrame]:
    """
    Generate a temporary MP4 demo video and a per-frame metrics DataFrame.
    """
    video_dir = tempfile.mkdtemp(prefix="drs_demo_")

    try:
        writer, video_path = _open_video_writer(
            video_dir=video_dir,
            fps=params.fps,
            size=(params.width, params.height),
        )

        rows: List[Dict[str, Any]] = []

        try:
            for frame_index in range(params.frames):
                frame_rgb, row = compose_frame(params, frame_index)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                rows.append(row)
        finally:
            writer.release()

        if not rows:
            raise RuntimeError("No frames were generated.")

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            raise RuntimeError("Video generation produced an empty file.")

        stable_path = os.path.join(video_dir, "demo.mp4")
        if video_path != stable_path:
            shutil.move(video_path, stable_path)
            video_path = stable_path

        return video_dir, video_path, pd.DataFrame(rows)

    except Exception:
        shutil.rmtree(video_dir, ignore_errors=True)
        raise


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Diabetic Retinopathy Screening Demo",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Diabetic Retinopathy Screening from Fundus Photographs")
    st.caption(
        "Synthetic Streamlit simulation for screening workflow demonstration. "
        "Not a medical device. Not for clinical use. No real patient data is used."
    )

    # Sidebar controls
    st.sidebar.header("Simulation controls")
    st.sidebar.caption(
        "Adjust the synthetic inputs, then press Regenerate video. "
        "The live preview updates immediately as you move controls."
    )

    eye = st.sidebar.selectbox("Eye", options=("Left", "Right"), index=0)

    severity = st.sidebar.slider(
        "Simulated lesion severity",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULTS["severity"]),
        step=0.01,
    )

    quality = st.sidebar.slider(
        "Simulated acquisition quality",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULTS["quality"]),
        step=0.01,
    )

    noise = st.sidebar.slider(
        "Sensor noise",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULTS["noise"]),
        step=0.01,
    )

    blur = st.sidebar.slider(
        "Focus blur",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULTS["blur"]),
        step=0.01,
    )

    exposure = st.sidebar.slider(
        "Exposure",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULTS["exposure"]),
        step=0.01,
    )

    simulate_progression = st.sidebar.checkbox(
        "Simulate progression over clip",
        value=bool(DEFAULTS["simulate_progression"]),
    )

    resolution = st.sidebar.selectbox(
        "Image size",
        options=(256, 320, 384, 512),
        index=2,
    )

    frames = st.sidebar.number_input(
        "Clip length (frames)",
        min_value=12,
        max_value=180,
        value=int(DEFAULTS["frames"]),
        step=6,
    )

    fps = st.sidebar.number_input(
        "Frame rate",
        min_value=2,
        max_value=30,
        value=int(DEFAULTS["fps"]),
        step=1,
    )

    seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=99999,
        value=int(DEFAULTS["seed"]),
        step=1,
    )

    st.sidebar.subheader("Decision thresholds")

    refer_threshold = st.sidebar.slider(
        "Refer threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULTS["refer_threshold"]),
        step=0.01,
    )

    review_threshold = st.sidebar.slider(
        "Review threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULTS["review_threshold"]),
        step=0.01,
    )

    min_quality = st.sidebar.slider(
        "Minimum acceptable quality",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULTS["min_quality"]),
        step=0.01,
    )

    regenerate = st.sidebar.button("Regenerate video", type="primary")

    refer_threshold = float(refer_threshold)
    review_threshold = min(float(review_threshold), refer_threshold)

    current_params = DemoParams(
        width=int(resolution),
        height=int(resolution),
        frames=int(frames),
        fps=int(fps),
        seed=int(seed),
        eye=str(eye),
        severity=float(severity),
        quality=float(quality),
        noise=float(noise),
        blur=float(blur),
        exposure=float(exposure),
        simulate_progression=bool(simulate_progression),
        refer_threshold=refer_threshold,
        review_threshold=review_threshold,
        min_quality=float(min_quality),
    )

    if "demo_ready" not in st.session_state:
        st.session_state.demo_ready = False

    if (not st.session_state.demo_ready) or regenerate:
        old_dir = st.session_state.get("video_dir")

        with st.spinner("Generating live demo video..."):
            try:
                video_dir, video_path, metrics_df = generate_demo_video(current_params)
            except Exception as exc:
                st.session_state.generation_error = str(exc)
                existing_path = st.session_state.get("video_path")
                if not existing_path or not os.path.exists(existing_path):
                    st.session_state.demo_ready = False
            else:
                if old_dir and old_dir != video_dir and os.path.isdir(old_dir):
                    shutil.rmtree(old_dir, ignore_errors=True)

                st.session_state.video_dir = video_dir
                st.session_state.video_path = video_path
                st.session_state.metrics_df = metrics_df
                st.session_state.params = current_params
                st.session_state.demo_ready = True
                st.session_state.pop("generation_error", None)

    generation_error = st.session_state.get("generation_error")
    if generation_error:
        st.error(
            f"Video generation failed: {generation_error}. "
            "The live preview may still work."
        )

    tab_video, tab_metrics, tab_explain, tab_about = st.tabs(
        [
            "Live demo video",
            "Metrics",
            "Explanation",
            "About / governance",
        ]
    )

    preview_row: Optional[Dict[str, Any]] = None

    with tab_video:
        video_path = st.session_state.get("video_path")

        if video_path and os.path.exists(video_path):
            st.subheader("Automatically generated demo clip")

            with open(video_path, "rb") as video_file:
                st.video(video_file.read())

            with open(video_path, "rb") as download_file:
                st.download_button(
                    label="Download demo video",
                    data=download_file.read(),
                    file_name=os.path.basename(video_path),
                    mime="video/mp4",
                )
        else:
            st.info(
                "No video file is currently available. "
                "The live preview below can still demonstrate interaction."
            )

        st.subheader("Live preview")
        st.caption(
            "This single-frame preview updates when you change sidebar controls. "
            "Press Regenerate video to create a new MP4 clip."
        )

        try:
            preview_frame, preview_row = compose_frame(current_params, 0)
            st.image(preview_frame, channels="RGB", use_container_width=True)

            with st.expander("Preview decision payload", expanded=False):
                st.json(preview_row)
        except Exception as exc:
            st.warning(f"Live preview failed: {exc}")

    with tab_metrics:
        metrics_df = st.session_state.get("metrics_df")

        if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            st.subheader("Per-frame simulation metrics")
            st.dataframe(metrics_df, use_container_width=True)

            chart_df = metrics_df.set_index("frame_index")
            st.line_chart(
                chart_df[
                    [
                        "quality_score",
                        "referable_probability",
                        "uncertainty",
                    ]
                ]
            )

            cols = st.columns(4)
            cols[0].metric("Frames", int(len(metrics_df)))
            cols[1].metric(
                "Refer flags",
                int((metrics_df["action"] == "Refer to ophthalmology").sum()),
            )
            cols[2].metric(
                "Review flags",
                int((metrics_df["action"] == "Clinician review").sum()),
            )
            cols[3].metric(
                "Repeat flags",
                int((metrics_df["action"] == "Repeat image").sum()),
            )

            st.subheader("Action distribution")
            st.bar_chart(metrics_df["action"].value_counts())
        else:
            st.info("Generate a video to populate the metrics table.")

    with tab_explain:
        st.subheader("Synthetic decision policy")
        st.markdown(
            """
            This demo uses transparent, non-clinical rules:

            1. **Quality gate**  
               If image quality is below the configured minimum, the action is  
               `Repeat image`.

            2. **Referral rule**  
               If the synthetic referable probability is above the refer threshold,  
               the action is `Refer to ophthalmology`.

            3. **Review rule**  
               If probability is borderline or uncertainty is high, the action is  
               `Clinician review`.

            4. **Routine follow-up**  
               Otherwise, the action is `Routine follow-up`.

            The heatmap is generated from synthetic lesion markers. It is not a
            validated model attribution method such as Grad-CAM, SHAP, or
            attention rollout.
            """
        )

        if preview_row is not None:
            st.subheader("Current preview payload")
            st.json(preview_row)
        else:
            st.info("Adjust controls in the Live demo video tab to populate this payload.")

    with tab_about:
        st.subheader("Intended use")
        st.markdown(
            """
            This application is a local simulation for demonstrating a possible
            diabetic retinopathy screening workflow. It is intended for education,
            prototyping, stakeholder discussion, and engineering validation planning.
            """
        )

        st.subheader("Explicit limitations")
        st.markdown(
            """
            - No real patient images are used.
            - No trained diagnostic model is included.
            - No clinical validation has been performed.
            - No regulatory clearance is implied.
            - Outputs must not be used for diagnosis, triage, or treatment decisions.
            - The clinician remains accountable for all real-world decisions.
            """
        )

        st.subheader("Risks and mitigations")
        st.markdown(
            """
            | Risk | Mitigation shown in this demo | Next step for real deployment |
            |---|---|---|
            | Overtrust in synthetic outputs | Persistent non-clinical disclaimer | Clinician training and UI governance |
            | Poor image quality leading to unsafe output | Quality gate and repeat-image action | Validated QC model and prospective monitoring |
            | Bias across devices/sites/populations | Configurable slices and metrics table | Multi-site validation and subgroup analysis |
            | Unclear accountability | Human-in-the-loop action labels | Signed clinical protocol and audit trail |
            | Privacy risk | Local synthetic data only | DPIA, de-identification, access controls |
            """
        )

        st.subheader("Suggested next engineering steps")
        st.markdown(
            """
            1. Replace synthetic renderer with approved de-identified dataset ingestion.
            2. Add DICOM/NIfTI metadata validation and patient-level split enforcement.
            3. Train a real classifier with versioned data, code, config, and seeds.
            4. Calibrate probabilities and define operating points with clinical owners.
            5. Evaluate on an independent external test set.
            6. Add out-of-distribution detection and deferral policy.
            7. Run silent-mode prospective pilot with clinician oversight.
            """
        )


if __name__ == "__main__":
    main()