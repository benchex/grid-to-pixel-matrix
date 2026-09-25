import cv2
import numpy as np
import sys
import os
import argparse
import re
import warnings

# Suppress Chainer/Mac Accelerate warning strings to keep terminal layout tidy
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CHAINER_WARN"] = "0"

def normalize_filename(filename):
    """
    Cleans up the base filename by replacing spaces with underscores 
    and removing any special non-alphanumeric characters.
    """
    clean = re.sub(r'[\s\-]+', '_', filename)
    clean = re.sub(r'[^\w]', '', clean)
    clean = re.sub(r'_+', '_', clean)
    return clean.strip('_').lower()

def detect_cell_size_by_autocorrelation(profile, min_cell=8, max_cell=60):
    """
    Finds the fundamental repeating pattern (cell size) in a 1D line profile
    using auto-correlation. This completely ignores compression noise blocks.
    """
    n = len(profile)
    p_norm = profile - np.mean(profile)
    
    best_lag = None
    max_correlation = -1
    
    for lag in range(min_cell, min(max_cell, n // 2)):
        correlation = np.sum(p_norm[:n-lag] * p_norm[lag:])
        if correlation > max_correlation:
            max_correlation = correlation
            best_lag = lag
            
    return best_lag

def run_inline_ai_denoise(img, noise_level=2):
    """
    Executes the local Waifu2x machine learning model directly in memory
    on the loaded image array to flatten compression artifacts using a specified strength.
    """
    print(f"🤖 Processing image through built-in Waifu2x AI models (Denoise Level: {noise_level})...")
    try:
        from waifu2x import Waifu2x
        from PIL import Image
        
        # Convert OpenCV BGR Image array to PIL RGB Image format
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Initialize model: apply user-defined noise reduction level, keep native scale (1x)
        model = Waifu2x(model_dir=None, gpus=-1) # -1 runs offline on CPU safely
        cleaned_pil = model.compile(pil_img, method='noise', noise_level=noise_level)
        
        # Convert back to native OpenCV BGR numpy array
        return cv2.cvtColor(np.array(cleaned_pil), cv2.COLOR_RGB2BGR)
    except ImportError:
        print("⚠️ Warning: 'waifu2x' library not found in this environment!")
        print("Please install it running: pip install waifu2x chainer")
        print("Proceeding with standard execution mapping...")
        return img
    except Exception as e:
        print(f"⚠️ Warning: AI engine failed due to an error: {e}")
        print("Proceeding with standard execution mapping...")
        return img

def convert_grid_to_pixel_art(image_path, output_arg, user_w, user_h, run_denoise=False, noise_level=2):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or read the file '{image_path}'.")
        return
        
    # Optional direct AI preprocessing block execution with adaptive noise tuning
    if run_denoise:
        img = run_inline_ai_denoise(img, noise_level=noise_level)
        
    img_h, img_w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Dynamically locate active boundaries by skipping solid empty borders if present
    non_black = np.where(gray > 15)
    
    if len(non_black[0]) == 0 or len(non_black[1]) == 0:
        min_y, max_y = 0, img_h - 1
        min_x, max_x = 0, img_w - 1
    else:
        min_y, max_y = np.min(non_black[0]), np.max(non_black[0])
        min_x, max_x = np.min(non_black[1]), np.max(non_black[1])
        
        # If content spans nearly the full size, snap directly to edges
        if (max_y - min_y) > img_h * 0.95: min_y, max_y = 0, img_h - 1
        if (max_x - min_x) > img_w * 0.95: min_x, max_x = 0, img_w - 1

    # 2. Extract global Sobel gradient profiles to expose grid line structures
    grad_x = np.abs(cv2.Sobel(gray[min_y:max_y+1, min_x:max_x+1], cv2.CV_64F, 1, 0, ksize=1))
    grad_y = np.abs(cv2.Sobel(gray[min_y:max_y+1, min_x:max_x+1], cv2.CV_64F, 0, 1, ksize=1))
    
    profile_x = np.sum(grad_x, axis=0) # Tracks vertical lines
    profile_y = np.sum(grad_y, axis=1) # Tracks horizontal lines
    
    # 3. Route parameter assignment (Auto-detection vs User-defined overrides)
    if user_w is not None and user_h is not None:
        grid_w, grid_h = user_w, user_h
        print(f"📋 Using manually specified grid parameters: {grid_w}x{grid_h}")
    else:
        print("🤖 Running auto-correlation pattern recognition scanner...")
        cell_w = detect_cell_size_by_autocorrelation(profile_x)
        cell_h = detect_cell_size_by_autocorrelation(profile_y)
        
        if cell_w and cell_h:
            grid_w = int(round((max_x - min_x + 1) / cell_w))
            grid_h = int(round((max_y - min_y + 1) / cell_h))
            print(f"✅ Auto-detected precise grid size: {grid_w}x{grid_h}")
        else:
            print("❌ Error: Pattern recognition failed to find a clean repeating grid pattern.")
            print("Please fallback manually using the --width and --height flags.")
            sys.exit(1)

    # 4. Map center point pixels cleanly using dynamic step arrays
    pixel_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    x_steps = np.linspace(min_x, max_x, grid_w + 1, dtype=int)
    y_steps = np.linspace(min_y, max_y, grid_h + 1, dtype=int)
    
    for row in range(grid_h):
        for col in range(grid_w):
            cx = (x_steps[col] + x_steps[col + 1]) // 2
            cy = (y_steps[row] + y_steps[row + 1]) // 2
            
            cx = min(max(0, cx), img_w - 1)
            cy = min(max(0, cy), img_h - 1)
            
            pixel_canvas[row, col] = img[cy, cx]
            
    # 5. DYNAMIC NAMING ENGINE: Construct output suffix based on flags used
    # Extract base directory and filename mapping safely
    file_base, _ = os.path.splitext(os.path.basename(output_arg if output_arg else image_path))
    dir_name = os.path.dirname(output_arg if output_arg else image_path)
    clean_base = normalize_filename(file_base)
    
    # Strip any older temporary trailing terms if present
    if clean_base.endswith('_pixel'):
        clean_base = clean_base[:-6]
        
    # Build flag descriptors string dynamically
    flag_suffix = ""
    if user_w is not None and user_h is not None:
        flag_suffix += f"_{grid_w}x{grid_h}"
    if run_denoise:
        flag_suffix += f"_denoise_n{noise_level}"
        
    # Append descriptors directly into final names map structures
    matrix_filename = f"{clean_base}{flag_suffix}_matrix.png"
    preview_filename = f"preview_{clean_base}{flag_suffix}_matrix.png"
    
    matrix_path = os.path.join(dir_name, matrix_filename) if dir_name else matrix_filename
    preview_path = os.path.join(dir_name, preview_filename) if dir_name else preview_filename
    
    # Save output files
    cv2.imwrite(matrix_path, pixel_canvas)
    print(f"\n✅ Successfully saved 1:1 matrix file for WLED to: {matrix_path}")

    preview_scale = 16  
    scaled_preview = cv2.resize(
        pixel_canvas, 
        (grid_w * preview_scale, grid_h * preview_scale), 
        interpolation=cv2.INTER_NEAREST
    )
    
    cv2.imwrite(preview_path, scaled_preview)
    print(f" Saved crisp human-viewable preview to:      {preview_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a grid-based template into clean pixel art.")
    parser.add_argument("input", nargs="?", help="Path to the input grid image file")
    parser.add_argument("-o", "--output", default=None, help="Path to the output image file (optional)")
    
    # User friendly descriptive flags
    parser.add_argument("-w", "--width", type=int, help="Grid width/columns in squares")
    parser.add_argument("-g", "--height", type=int, dest="height", help="Grid height/rows in squares")
    
    # Integrated optional inline AI flag toggle
    parser.add_argument("-d", "--denoise", action="store_true", help="Run inline local Waifu2x AI engine to wipe out JPEG noise blocks")
    # Added variable noise slider control modifier
    parser.add_argument("-n", "--noise", type=int, choices=[0, 1, 2, 3], default=2, help="Set the AI denoising strength level (0=Low, 3=Maximum. Default is 2)")
    
    args = parser.parse_args()
    
    if not args.input:
        print("❌ Error: Missing required image path parameter!")
        print("Usage: python3 converter.py <image_path> [--width <width> --height <height>] [--denoise] [--noise 0-3]")
        sys.exit(1)
        
    convert_grid_to_pixel_art(args.input, args.output, args.width, args.height, args.denoise, args.noise)
