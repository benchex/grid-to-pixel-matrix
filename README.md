# Grid to Pixel Matrix Converter 👾

A precise, lightweight Python command-line utility designed to convert grid-based template images into flawless, borderless 1:1 pixel art files. It is explicitly optimized to prepare clean source graphics for **WLED panels**, **DIY LED matrices**, and retro game engines.

Unlike standard image downscalers or resizing tools that blur borders and create muddy colors, this script utilizes **pinpoint center-coordinate sampling** to entirely isolate individual grid colors, completely stripping away grid lines and compression artifacts.

---

## 🚀 Features

- **True 1:1 Pixel Mapping:** Outputs a raw, lossless `.png` matching your exact matrix resolution (e.g., 32x32 pixels).
- **Zero Grid Bleeding:** Ignores black grid borders and JPEG compression grime by mathematically targeting the dead-center of every single cell.
- **Automatic Preview Generation:** Automatically saves a high-contrast, upscaled visual preview using `Nearest Neighbor` interpolation so you can check your work on a standard monitor without blurring.
- **Fully Customizable:** Adapts instantly to any irregular or non-square aspect ratio grid dimensions via command-line arguments.

---

## 📦 Installation

This script requires Python 3 and the **OpenCV** library.

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

Because every source grid asset is built differently, you must pass the exact horizontal grid count (`-w`) and vertical grid count (`-g`).

> 💡 **Supported Input Formats:** This script natively processes any standard image format handled by OpenCV, including `.png`, `.jpg`, `.jpeg`, `.webp`, and `.bmp`. Using `.png` inputs is highly recommended to minimize color distortion caused by compression.

```bash
python3 pixelart.py <input_image_path> -w <grid_width> -g <grid_height> [-o <output_name>]
```

### Example Usage
If your input template image is an image asset containing a 32x32 layout:

```bash
python3 pixelart.py my_template.png -w 32 -g 32
```

### Output Files Produced:
- **`my_template_pixel.png`**: The raw **32x32 pixel** image. This is the tight, lossless matrix file you upload straight to your WLED controller or PixelForge layout.
- **`preview_my_template_pixel.png`**: A crisp, cleanly upscaled (16x multiplier) version to easily view, share, or store on your computer.

---

## 📐 How the Math Works

When standard image resizing algorithms (`BILINEAR`, `BICUBIC`) handle grid graphics, they average pixels together. This pulls the black grid borders into the color squares, resulting in a dark, stained, or heavily artifacted final matrix.

This utility completely bypasses resizing arrays:
1. It calculates the exact decimal pixel step values (`image_width / grid_columns`).
2. It tracks down the pinpoint floating-point coordinates for the dead center of every single independent square cell (e.g., cell index `+ 0.5`).
3. It takes a single, isolated color value reading right from that coordinate matrix index and drops it directly onto a brand-new canvas.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
