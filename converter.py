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

def find_grid_line_coordinates(img_gray, min_bound, max_bound, is_horizontal=True, min_cell_size=15):
    """
    Scans the active area to map the exact coordinate of every single grid line
    by tracking local contrast dips. This creates an adaptive spatial map.
    """
    max_bound - min_bound
    mid = (min_bound + max_bound) // 2
    if is_horizontal:
        profile = np.mean(img_gray[:, max(0, mid-20):min(img_gray.shape, mid+20)], axis=1)
    else:
        profile = np.mean(img_gray[max(0, mid-20):min(img_gray.shape, mid+20), :], axis=0)
        
    lines = [min_bound]
    
    i = min_bound + 5
    while i < max_bound - 5:
        if profile[i] < profile[i-2] - 3 and profile[i] < profile[i+2] - 3:
            if (i - lines[-1]) >= min_cell_size:
                lines.append(i)
                i += 5
                continue
        i += 1
        
    lines.append(max_bound)
    return lines

def convert_grid_to_pixel_art(image_path, output_path, user_w, user_h):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or read the file '{image_path}'.")
        return
        
    img_h, img_w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Dynamically find the active grid bounding box (strip away outer solid padding)
    non_black = np.where(gray > 15)
    if len(non_black) == 0 or len(non_black) == 0:
        print("❌ Error: Image appears to be completely black.")
        sys.exit(1)
        
    min_y, max_y = np.min(non_black), np.max(non_black)
    min_x, max_x = np.min(non_black), np.max(non_black)
    
    # 2. Build an adaptive spatial coordinate map of the grid line boundaries
    print("🤖 Analyzing image grid structural boundaries...")
    x_lines = find_grid_line_coordinates(gray, min_x, max_x, is_horizontal=False)
    y_lines = find_grid_line_coordinates(gray, min_y, max_y, is_horizontal=True)
    
    grid_w = len(x_lines) - 1
    grid_h = len(y_lines) - 1
    
    # 3. Route parameters (Allow manual overrides via long-form word flags)
    if user_w is not None and user_h is not None:
        grid_w, grid_h = user_w, user_h
        print(f"📋 Using manually specified grid parameters: {grid_w}x{grid_h}")
        x_steps = np.linspace(min_x, max_x, grid_w + 1, dtype=int)
        y_steps = np.linspace(min_y, max_y, grid_h + 1, dtype=int)
    else:
        print(f"✅ Auto-detected precise grid size: {grid_w}x{grid_h}")
        x_steps = x_lines
        y_steps = y_lines

    # 4. Map the center points cleanly using our precise coordinate map boundaries
    pixel_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for row in range(grid_h):
        for col in range(grid_w):
            cx = (x_steps[col] + x_steps[col + 1]) // 2
            cy = (y_steps[row] + y_steps[row + 1]) // 2
            
            cx = min(max(0, cx), img_w - 1)
            cy = min(max(0, cy), img_h - 1)
            
            pixel_canvas[row, col] = img[cy, cx]
            
    # Output file paths setup
    file_base, _ = os.path.splitext(os.path.basename(output_path))
    dir_name = os.path.dirname(output_path)
    clean_base = normalize_filename(file_base)
    
    if clean_base.endswith('_pixel'):
        clean_base = clean_base[:-6]
        
    matrix_filename = f"{clean_base}_matrix.png"
    preview_filename = f"preview_{clean_base}_matrix.png"
    
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
    parser.add_argument("-o", "--output", help="Path to the output image file (optional)")
    
    # UPGRADED PARAMETERS: Descriptive long-form word flags for beginners
    # Short aliases (-w, -g) are kept optionally active, but custom dest targets the words directly
    parser.add_argument("-w", "--width", type=int, help="Grid width/columns in squares")
    parser.add_argument("-g", "--height", type=int, dest="height", help="Grid height/rows in squares")
    
    args = parser.parse_args()
    
    if not args.input:
        print("❌ Error: Missing required image path parameter!")
        print("Usage: python3 converter.py <image_path> [--width <width> --height <height>]")
        sys.exit(1)
        
    if args.output:
        output_file = args.output
    else:
        file_base, _ = os.path.splitext(args.input)
        output_file = f"{file_base}_pixel.png"
        
    convert_grid_to_pixel_art(args.input, output_file, args.width, args.height)
