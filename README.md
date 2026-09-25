# Grid to Pixel Matrix Converter 👾

A precise, lightweight Python command-line utility designed to convert grid-based pixel art templates into flawless, borderless 1:1 pixel files. It is explicitly optimized to prepare clean source graphics for **WLED displays**, **DIY LED matrices**, and retro game engines.

Unlike standard image downscalers or resizing tools that blur borders and create muddy colors, this script utilizes an **adaptive auto-correlation line-mapper** to automatically map grid geometry, then relies on **pinpoint center-coordinate sampling** to entirely isolate pure pixel colors. It also features a **built-in local AI denoising engine** via Waifu2x to iron out compression grime on low-quality files.

---

## 🚀 Features

- **True 1:1 Pixel Mapping:** Outputs a raw, lossless `.png` matching your exact hardware matrix setup (e.g., 24x24 pixels).
- **Intelligent Auto-Detection:** Automatically scans the asset canvas using auto-correlation pattern recognition to map line profiles and cell variations—completely hands-free.
- **Optional Inline AI Denoising:** Pass an optional flag to execute a local Waifu2x neural network model directly in memory to flatten blocky JPEG artifacts before mapping.
- **Flag-Adaptive Naming:** Automatically appends active configuration parameters (dimensions, noise levels) to output filenames to prevent accidental overwrites during testing.
- **Zero Grid Bleeding:** Targets the dead-center coordinates of every independent square cell to bypass gridlines or lossy compression grime entirely.
- **Automatic Preview Export:** Saves a crisp, high-contrast upscaled version using sharp `Nearest Neighbor` interpolation to easily review on a standard PC monitor without blurring.

---

## 📦 Installation

This script requires Python 3, **OpenCV**, and optionally **Waifu2x** for deep-learning artifact removal.

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd grid-to-pixel-matrix
   ```

2. **Install core dependencies & AI models:**
   ```bash
   pip install opencv-python numpy waifu2x chainer
   ```

---

## 🛠️ Usage

This utility natively handles standard image formats, including `.png`, `.jpg`, `.jpeg`, and `.webp`.

```bash
python3 converter.py <input_image_path> [--width <grid_width> --height <grid_height>] [--denoise] [--noise 0-3] [-o <output_name>]
```

### 🤖 Option A: Hands-Free Auto-Detection (Default / Recommended)
Let the script trace structural boundaries and map cell profiles automatically:
```bash
python3 converter.py pixelart.jpg
```

### 🧠 Option B: Dynamic AI Denoise (For Muddy JPEGs)
Run an embedded Waifu2x machine learning filter to wipe out blocky compression textures in-memory before conversion (Default noise strength is 2):
```bash
python3 converter.py pixelart.jpg --denoise --noise 2
```

### 📋 Option C: Explicit Manual Override
Bypass automated calculations completely by specifying the exact matrix grid dimensions:
```bash
python3 converter.py pixelart.jpg --width 24 --height 24
```

### Output Files Produced:
- **`pixelart_matrix.png`** (or labeled with active flags like `_denoise_n2_matrix.png`): The raw, tiny uncompressed matrix pixel image file. This is the asset you upload straight to your WLED control dashboard.
- **`preview_pixelart_matrix.png`**: A crisp, cleanly upscaled (16x scaling multiplier) companion copy to easily view, store, or share on a regular monitor.

---

## ⚖️ Performance Tradeoffs: When to use AI Denoising

AI preprocessing is a double-edged sword when working with low-resolution matrix grids. Because physical LED panels rely on high-contrast, razor-sharp color transitions to look good to the human eye, review the following guidelines before deploying flags:

| Setting | 🟢 Pros | 🔴 Cons | Best Used For |
| :--- | :--- | :--- | :--- |
| **Standard Mode** (No AI) | Preserves maximum **sharpness** and deliberate shading details on character sprites. | Background fields can retain blocky JPEG compression artifacts. | High-quality templates or crisp source image files. |
| **Denoise Mode** (`--denoise`) | Perfectly **flattens** noisy, compressed backgrounds into unified solid shades. | Can slightly blur micro-details or soften intentional pixel color steps on smaller features. | Heavily squashed JPEG files where compression noise is disrupting the layout scanners. |

---

## 📐 How the Math Works

When standard image resizing algorithms (`BILINEAR`, `BICUBIC`) handle grid graphics, they average neighbor pixels together. This pulls the blurry grid borders into the color squares, resulting in a dark, stained, or heavily artifacted final matrix display.

This utility completely bypasses resizing filters:
1. If `--denoise` is flag-active, it passes the source image through a local, offline machine learning port to reconstruct squashed structural details.
2. It analyzes global Sobel gradient profiles across structural strips to build a cross-correlation profile, determining the fundamental repeating pattern (cell size).
3. It maps pinpoint floating-point coordinates for the dead center of every single independent square cell (e.g., cell index `+ 0.5`).
4. It reads a single color value directly from that center coordinate and drops it directly onto a brand-new canvas.

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)** - see the [LICENSE](LICENSE) file for details.
