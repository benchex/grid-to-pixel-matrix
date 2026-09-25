# Grid to Pixel Matrix Converter 👾

A precise, lightweight Python command-line utility designed to convert grid-based pixel art templates into flawless, borderless 1:1 pixel files. It is explicitly optimized to prepare clean source graphics for **WLED displays**, **DIY LED matrices**, and retro game engines.

Unlike standard image downscalers or resizing tools that blur borders and create muddy colors, this script utilizes a **top-left cell contour scanner** to automatically calculate grid geometry, then relies on **pinpoint center-coordinate sampling** to entirely isolate pure pixel colors.

---

## 🚀 Features

- **True 1:1 Pixel Mapping:** Outputs a raw, lossless `.png` matching your exact hardware matrix setup (e.g., 27x30 pixels).
- **Intelligent Auto-Detection:** Automatically scans the top-left corner of the asset file to calculate cell boundaries and matrix ratios—completely hands-free.
- **Zero Grid Bleeding:** Targets the dead-center coordinates of every independent square to bypass black gridlines or lossy compression grime entirely.
- **Automatic Preview Export:** Saves a crisp, high-contrast upscaled version using sharp `Nearest Neighbor` interpolation to easily review on a standard PC monitor without blurring.

---

## 📦 Installation

This script requires Python 3 and the **OpenCV** image processing library.

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
python3 converter.py <input_image_path> [-w <grid_width> -g <grid_height>] [-o <output_name>]
```

### 🤖 Option A: Hands-Free Auto-Detection (Recommended)
Let the script trace the edge constraints and calculate the grid sizing rules automatically:

```bash
python3 converter.py pixelart.png
```

### 📋 Option B: Manual Override Flag
If an asset template has missing outer grid lines, excessive padding, or dense custom banners, you can override the detector and force specific matrix parameters:

```bash
python3 converter.py pixelart.png -w 27 -g 30
```

### Output Files Produced:
- **`pixelart_matrix.png`**: The raw, tiny uncompressed matrix pixel image file. This is the asset you upload straight to your WLED control dashboard.
- **`preview_pixelart_matrix.png`**: A crisp, cleanly upscaled (16x scaling multiplier) companion copy to easily view, store, or share.

---

## 📐 How the Math Works

When standard image resizing algorithms (`BILINEAR`, `BICUBIC`) handle grid graphics, they average neighbor pixels together. This pulls the black grid borders into the color squares, resulting in a dark, stained, or heavily artifacted final matrix.

This utility completely bypasses resizing filters:
1. It analyzes shapes in the top-left quadrant of the image to isolate the pixel area of a single block.
2. It divides total dimensions by that layout block to extrapolate total columns and rows.
3. It maps pinpoint floating-point coordinates for the dead center of every single independent square cell (e.g., cell index `+ 0.5`).
4. It reads a single color value directly from that coordinate index and drops it directly onto a brand-new canvas.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
