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
    clean = re.sub(r'[\s\-]+', '_', filename)
    clean = re.sub(r'[^\w]', '', clean)
    clean = re.sub(r'_+', '_', clean)
    return clean.strip('_').lower()

def detect_grid_by_top_left_cell(img):
    """
    Finds the first real grid cell in the top-left corner by isolating 
    the bounding box edges, ignoring details in the center of the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. Clean the image to isolate dark grid borders
    # Standard thresholding to turn dark lines into white lines on a black canvas
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Find shapes (contours) in the top-left quarter of the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    possible_widths = []
    possible_heights = []
    
    for cnt in contours:
        x, y, box_w, box_h = cv2.boundingRect(cnt)
        
        # Only look at shapes in the top-left region of the image
        if x < w // 3 and y < h // 3:
            # Grid cells must be reasonably sized (larger than noise, smaller than half the image)
            if 10 < box_w < (w // 4) and 10 < box_h < (h // 4):
                # Check if it's roughly square-shaped
                if 0.8 < (box_w / box_h) < 1.2:
                    possible_widths.append(box_w)
                    possible_heights.append(box_h)
                    
    # 3. Use the most common bounding box size to determine cell dimensions
    if possible_widths and possible_heights:
        cell_w = np.median(possible_widths)
        cell_h = np.median(possible_heights)
        
        # Calculate total grid slots based on the detected corner cell size
        grid_w = int(round(w / cell_w))
        grid_h = int(round(h / cell_h))
        return grid_w, grid_h
        
    return None, None

def convert_grid_to_pixel_art(image_path, output_path, user_w, user_h):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or read the file '{image_path}'.")
        return
        
    img_h, img_w, _ = img.shape
    
    # Target grid dimensions setup
    if user_w is not None and user_h is not None:
        grid_w, grid_h = user_w, user_h
        print(f"📋 Using manually specified grid parameters: {grid_w}x{grid_h}")
    else:
        print("🤖 Tracking edge bounds to find the first complete grid cell...")
        grid_w, grid_h = detect_grid_by_top_left_cell(img)
        
        if grid_w is None or grid_h is None:
            print("❌ Error: Boundary scan could not detect solid grid borders automatically.")
            print("Please override manually using the -w and -g flags.")
            sys.exit(1)
        print(f"✅ Auto-detected grid size matrix: {grid_w}x{grid_h}")
    
    # Safe output naming setups
    file_base, _ = os.path.splitext(os.path.basename(output_path))
    dir_name = os.path.dirname(output_path)
    clean_base = normalize_filename(file_base)
    
    if clean_base.endswith('_pixel'):
        clean_base = clean_base[:-6]
        
    matrix_filename = f"{clean_base}_matrix.png"
    preview_filename = f"preview_{clean_base}_matrix.png"
    
    matrix_path = os.path.join(dir_name, matrix_filename) if dir_name else matrix_filename
    preview_path = os.path.join(dir_name, preview_filename) if dir_name else preview_filename
    
    # Calculate stepping intervals
    cell_w = img_w / grid_w
    cell_h = img_h / grid_h
    
    # Process matrix canvas
    pixel_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for row in range(grid_h):
        for col in range(grid_w):
            center_x = int((col + 0.5) * cell_w)
            center_y = int((row + 0.5) * cell_h)
            
            center_x = min(max(0, center_x), img_w - 1)
            center_y = min(max(0, center_y), img_h - 1)
            
            pixel_canvas[row, col] = img[center_y, center_x]
            
    # Save output assets
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
    parser.add_argument("-o", "--output", help="Path to the output image file (optional)")
    parser.add_argument("-w", "--width", type=int, help="Grid width in squares (Optional if auto-detecting)")
    parser.add_argument("-g", "--height", type=int, help="Grid height in squares (Optional if auto-detecting)")
    
    args = parser.parse_args()
    
    if not args.input:
        print("❌ Error: Missing required image path parameter!")
        print("Usage: python3 converter.py <image_path> [-w <width> -g <height>]")
        sys.exit(1)
        
    if args.output:
        output_file = args.output
    else:
        file_base, _ = os.path.splitext(args.input)
        output_file = f"{file_base}_pixel.png"
        
    convert_grid_to_pixel_art(args.input, output_file, args.width, args.height)
