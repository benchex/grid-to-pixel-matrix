# Grid to Pixel Matrix Converter 👾

A precise, lightweight Python command-line utility designed to convert grid-based pixel art templates into flawless, borderless 1:1 pixel files. It is explicitly optimized to prepare clean source graphics for **WLED displays**, **DIY LED matrices**, and retro game engines.

Unlike standard image downscalers or resizing tools that blur borders and create muddy colors, this script utilizes an **adaptive contrast line-mapper** to automatically map grid geometry, then relies on **pinpoint center-coordinate sampling** to entirely isolate pure pixel colors.

---

## 🚀 Features

- **True 1:1 Pixel Mapping:** Outputs a raw, lossless `.png` matching your exact hardware matrix setup (e.g., 32x32 pixels).
- **Intelligent Auto-Detection:** Automatically scans the asset canvas to map line profiles and cell variations—completely hands-free.
- **Zero Grid Bleeding:** Targets the dead-center coordinates of every independent square cell to bypass gridlines or lossy compression grime entirely.
- **Automatic Preview Export:** Saves a crisp, high-contrast upscaled version using sharp `Nearest Neighbor` interpolation to easily review on a standard PC monitor without blurring.

---

## 📦 Installation

This script requires Python 3 and the **OpenCV** image processing library wrapper.

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd grid-to-pixel-matrix
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python numpy
   ```

---

## 🛠️ Usage

This utility natively handles standard image formats, including `.png`, `.jpg`, `.jpeg`, and `.webp`.

```bash
python3 converter.py <input_image_path> [--width <grid_width> --height <grid_height>] [-o <output_name>]
```

### 🤖 Option A: Hands-Free Auto-Detection (Recommended)
Let the script trace the structural edge boundaries and map cell profiles automatically:

```bash
python3 converter.py pixelart.png
```

### 📋 Option B: Explicit Manual Override
If an asset template has highly damaged outer lines, excessive banner data, or massive custom layouts, you can bypass the automated decision loop completely by typing your exact matrix dimensions:

```bash
python3 converter.py pixelart.png --width 32 --height 32
```

### Output Files Produced:
- **`pixelart_matrix.png`**: The raw, tiny uncompressed matrix pixel image file. This is the asset you upload straight to your WLED control dashboard.
- **`preview_pixelart_matrix.png`**: A crisp, cleanly upscaled (16x scaling multiplier) companion copy to easily view, store, or share.

---

## 📐 How the Math Works

When standard image resizing algorithms (`BILINEAR`, `BICUBIC`) handle grid graphics, they average neighbor pixels together. This pulls the blurry grid borders into the color squares, resulting in a dark, stained, or heavily artifacted final matrix display.

This utility completely bypasses resizing filters:
1. It analyzes average color profiles across structural strips to build a localized coordinate list of where every grid line lives.
2. It calculates the absolute geometric midpoint between adjacent grid lines on a square-by-square basis.
3. This completely prevents rounding errors and layout drifts on compressed or uneven assets.
4. It reads a single color value directly from that center coordinate and drops it directly onto a brand-new canvas.

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. 

### What this means:
- 🔓 **Reciprocity:** Anyone can use, modify, and distribute your code for free.
- 📦 **Copyleft Protection:** If anyone modifies this converter or integrates it into another software project, **their entire project must also be open-sourced under the exact same GPL-3.0 license.** Closed-source corporate commercialization of this tool is strictly prohibited.

See the [LICENSE](LICENSE) file for the full legal text.
