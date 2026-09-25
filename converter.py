import cv2
import numpy as np
import sys
import os
import argparse
import re

def normalize_filename(filename):
    """
    Cleans up the base filename by replacing spaces with underscores 
    and removing any special non-alphanumeric characters.
    """
    # Replace spaces and dashes with underscores
    clean = re.sub(r'[\s\-]+', '_', filename)
    # Remove any other strange characters except alphanumeric and underscores
    clean = re.sub(r'[^\w]', '', clean)
    # Strip double underscores if they happened during replacement
    clean = re.sub(r'_+', '_', clean)
    return clean.strip('_').lower()

def convert_grid_to_pixel_art(image_path, output_path, grid_w, grid_h):
    # 1. Load the original high-resolution grid image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or read the file '{image_path}'.")
        return
        
    img_h, img_w, _ = img.shape
    print(f"Loaded image size: {img_w}x{img_h} pixels.")
    print(f"Processing target grid: {grid_w}x{grid_h} blocks.")
    
    # 2. Extract directories and base filenames safely
    file_base, _ = os.path.splitext(os.path.basename(output_path))
    dir_name = os.path.dirname(output_path)
    
    # Clean the name to get rid of spaces, dashes, and double underscores
    clean_base = normalize_filename(file_base)
    
    # Strip trailing '_pixel' if it came from the automatic default naming fallback
    if clean_base.endswith('_pixel'):
        clean_base = clean_base[:-6]
        
    # Standardized output naming paths
    matrix_filename = f"{clean_base}_matrix.png"
    preview_filename = f"preview_{clean_base}_matrix.png"
    
    matrix_path = os.path.join(dir_name, matrix_filename) if dir_name else matrix_filename
    preview_path = os.path.join(dir_name, preview_filename) if dir_name else preview_filename
    
    # 3. Calculate the exact fractional size of each cell
    cell_w = img_w / grid_w
    cell_h = img_h / grid_h
    
    # 4. Initialize a brand new canvas matching the exact grid size
    pixel_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # 5. Pinpoint sample the exact center of every single grid square
    for row in range(grid_h):
        for col in range(grid_w):
            center_x = int((col + 0.5) * cell_w)
            center_y = int((row + 0.5) * cell_h)
            
            # Bound check safety boundaries
            center_x = min(max(0, center_x), img_w - 1)
            center_y = min(max(0, center_y), img_h - 1)
            
            # Extract exactly ONE pixel from the original image center point
            pixel_canvas[row, col] = img[center_y, center_x]
            
    # 6. Save the pure, raw 1:1 pixel art matrix (lossless PNG)
    cv2.imwrite(matrix_path, pixel_canvas)
    print(f"\n✅ Successfully saved 1:1 matrix file for WLED to: {matrix_path}")

    # 7. Generate a crisp, scaled-up file for easy computer viewing
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
    parser.add_argument("-o", "--output", help="Path to the output image file (optional)")
    parser.add_argument("-w", "--width", type=int, help="Grid width in squares (Required)")
    parser.add_argument("-g", "--height", type=int, help="Grid height in squares (Required)")
    
    args = parser.parse_args()
    
    # Safety Check: Catches empty inputs instantly to prevent terminal freezes
    if not args.input or args.width is None or args.height is None:
        print("❌ Error: Missing required command parameters!")
        print("Usage: python3 converter.py <image_path> -w <width> -g <height>")
        print("\nExample Run:")
        print("  python3 converter.py \"my pixelart file.jpeg\" -w 27 -g 30")
        sys.exit(1)
        
    # VS Code Fix: Handle fallback naming safely here to ensure output_file is never None
    if args.output:
        output_file = args.output
    else:
        file_base, _ = os.path.splitext(args.input)
        output_file = f"{file_base}_pixel.png"
        
    convert_grid_to_pixel_art(args.input, output_file, args.width, args.height)
