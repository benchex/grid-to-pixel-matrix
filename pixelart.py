import cv2
import numpy as np
import sys
import os
import argparse

def convert_grid_to_pixel_art(image_path, output_path, grid_w, grid_h):
    # 1. Load the original high-resolution grid image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or read the file '{image_path}'.")
        return
        
    img_h, img_w, _ = img.shape
    print(f"Loaded image size: {img_w}x{img_h} pixels.")
    print(f"Generating WLED Native Grid: {grid_w}x{grid_h} Matrix.")
    
    # 2. Calculate the exact fractional size of each cell
    cell_w = img_w / grid_w
    cell_h = img_h / grid_h
    
    # 3. Initialize a brand new canvas matching the exact grid size
    pixel_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # 4. Pinpoint sample the exact center of every single grid square
    for row in range(grid_h):
        for col in range(grid_w):
            center_x = int((col + 0.5) * cell_w)
            center_y = int((row + 0.5) * cell_h)
            
            # Bound check safety boundaries
            center_x = min(max(0, center_x), img_w - 1)
            center_y = min(max(0, center_y), img_h - 1)
            
            # Extract exactly ONE pixel from the original image center point
            pixel_canvas[row, col] = img[center_y, center_x]
            
    # FORCE PNG EXTENSION FOR THE RAW MATRIX IMAGE
    file_base, _ = os.path.splitext(output_path)
    wled_png_path = f"{file_base}.png"
            
    # 5. Save the pure, raw 1:1 pixel art image (27x30 pixels total)
    cv2.imwrite(wled_png_path, pixel_canvas)
    print(f"\n✅ Successfully saved 1:1 matrix file for WLED to: {wled_png_path}")

    # 6. Generate a crisp, scaled up file for computer preview/sharing
    preview_scale = 16  
    scaled_preview = cv2.resize(
        pixel_canvas, 
        (grid_w * preview_scale, grid_h * preview_scale), 
        interpolation=cv2.INTER_NEAREST
    )
    
    dir_name = os.path.dirname(wled_png_path)
    preview_filename = "preview_" + os.path.basename(wled_png_path)
    preview_path = os.path.join(dir_name, preview_filename) if dir_name else preview_filename
    
    cv2.imwrite(preview_path, scaled_preview)
    print(f" Saved crisp human-viewable preview to:       {preview_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a grid-based template into clean pixel art.")
    parser.add_argument("input", help="Path to the input grid image file")
    parser.add_argument("-o", "--output", help="Path to the output image file (optional)")
    parser.add_argument("-w", "--width", type=int, help="Grid width in squares (Required)")
    parser.add_argument("-g", "--height", type=int, help="Grid height in squares (Required)")
    
    args = parser.parse_args()
    
    if args.width is None or args.height is None:
        print("❌ Error: Missing grid dimensions!")
        print("Specify the layout size using the -w (width) and -g (height) parameters.")
        print("\nRun it exactly like this for your Wall-E image:")
        print(f"  python3 {os.path.basename(sys.argv)} \"{args.input}\" -w 27 -g 30")
        sys.exit(1)
        
    if args.output:
        output_file = args.output
    else:
        file_base, _ = os.path.splitext(args.input)
        output_file = f"{file_base}_pixel.png"
        
    convert_grid_to_pixel_art(args.input, output_file, args.width, args.height)
