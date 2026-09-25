# Grid to Pixel Matrix Converter 👾

A precise, lightweight Python command-line utility designed to convert grid-based pixel art templates into flawless, borderless 1:1 pixel files. It is explicitly optimized to prepare clean source graphics for **WLED displays**, **DIY LED matrices**, and retro game engines.

Unlike standard image downscalers or resizing tools that blur lines and mix adjacent colors, this script utilizes a global **adaptive auto-correlation line-mapper** to automatically discover grid geometry. It then uses **pinpoint coordinate center-sampling** or an area-consensus sieve to completely strip away grid lines and digital noise.

---

## 🚀 Features

- **True 1:1 Pixel Mapping:** Outputs a raw, lossless `.png` matching your exact hardware matrix setup (e.g., 24x24 or 32x32 pixels).
- **Intelligent Auto-Detection:** Automatically scans template canvases using cross-correlation signals to map line intervals—completely hands-free.
- **Inner-Cell Consensus Sieve (`--sieve`):** Aggregates every pixel inside an individual cell block and extracts the mathematical mode (most common color). This entirely flattens textured backgrounds and artifacts without rounding or blurring character details.
- **Optional Inline AI Denoising (`--denoise`):** Hooks natively into an embedded local Waifu2x neural network port to reconstruct highly broken lines on low-quality web saves in-memory.
- **Flag-Adaptive Naming:** Automatically stamps active terminal choices (e.g., `_24x24_sieve_matrix.png`) straight onto output assets to prevent file overwrites during calibration sweeps.

---

## 📦 Installation

This script requires Python 3 and the **OpenCV** image processing library wrapper.

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd grid-to-pixel-matrix
   ```

2. **Install core dependencies:**
   ```bash
   pip install opencv-python numpy waifu2x chainer
   ```

---

## 🛠️ Usage

This utility natively handles standard image formats, including `.png`, `.jpg`, `.jpeg`, and `.webp`.

```bash
python3 converter.py <input_image_path> [--width <grid_width> --height <grid_height>] [--sieve] [--denoise] [--noise 0-3]
```

### 🤖 Option A: Hands-Free Auto-Detection (Default)
Let the script trace structural boundaries and map cell profiles automatically using center-point sampling:
```bash
python3 converter.py character.png
```

### 🧹 Option B: Consensus Sieve Processing (Best Visual Results)
Wipes out all JPEG noise blocks and patchy background textures while keeping character lines razor-sharp:
```bash
python3 converter.py character.jpg --sieve
```

### 📋 Option C: Explicit Manual Override
Bypass automated calculations completely by specifying the exact matrix grid dimensions:
```bash
python3 converter.py character.png --width 24 --height 24
```

### Output Assets Produced:
- **`character_matrix.png`** (or flag-labeled like `_sieve_matrix.png`): The raw 1:1 matrix pixel file uploaded right to your WLED control dashboard.
- **`preview_character_matrix.png`**: A crisp, cleanly upscaled (16x multiplier) copy to easily view, manage, or share on a regular monitor.

---

## ⚖️ Processing Modes Guide

Choose the ideal processing workflow depending on the quality of your source template:

| Mode Flag | 🛠️ Under the Hood | 🟢 Best For | ❌ Avoid If |
| :--- | :--- | :--- | :--- |
| **Standard Mode** <br>*(Default)* | **Point-samples** the exact mathematical center of each calculated cell box. | Lossless `.png` sprites or high-quality templates where every pixel is clean. | The image background has messy, patchy compression textures. |
| **Consensus Sieve** <br>`--sieve` | Takes a **majority color vote** across the entire inner cell area, ignoring grid lines. | Flattening patchy background noise while keeping sharp character outlines. | The character has ultra-thin features (like a 1-pixel mouth) that might lose the vote. |
| **AI Denoise** <br>`--denoise` | Passes the image through a local **Waifu2x neural network** to repair line paths. | Heavily squashed JPEGs where the auto-detector fails to calculate the grid size. | You want to preserve exact, razor-sharp pixel shading inside the artwork. |

---

## 💡 Troubleshooting Outliers (Photographed Templates)

This script assumes a uniform, mathematically straight linear grid. If you attempt to process a template that is a **photograph of a physical book** or an image with **warped perspective lines** (such as a skewed camera shot), linear mapping arrays will naturally drift, hitting the gridlines.

For warped perspective images, you can bypass the script entirely and force your computer to handle the pixel consensus pooling by running this one-line resizing bypass in your terminal to instantly generate your 1:1 hardware matrix file:

```bash
python3 -c "import cv2; img=cv2.imread('warped_photo.jpg'); cv2.imwrite('clean_matrix.png', cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA))"
```

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)** - see the [LICENSE](LICENSE) file for details. Closed-source commercialization of this tool is strictly prohibited.
