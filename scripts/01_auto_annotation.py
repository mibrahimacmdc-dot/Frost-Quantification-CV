import os
import cv2
import numpy as np
import glob

# The two folders to exclude because of different camera setup
EXCLUDE_FOLDERS = ["13-sep-2022", "Labeledf_8-mar-2023"]

def process_folder(folder_path, output_base_dir):
    folder_name = os.path.basename(folder_path)
    if folder_name in EXCLUDE_FOLDERS:
        print(f"Skipping {folder_name} (excluded)")
        return

    # Find the reference image (0.png or 0.jpg)
    ref_paths = glob.glob(os.path.join(folder_path, "0.*"))
    if not ref_paths:
        print(f"No reference image (0.png/jpg) found in {folder_name}")
        return
    
    ref_path = ref_paths[0]
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        return
    
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    
    # Create output directory for this session's annotations
    out_dir = os.path.join(output_base_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Process all other images
    image_paths = glob.glob(os.path.join(folder_path, "*.*"))
    for img_path in image_paths:
        if img_path == ref_path:
            continue
            
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Image Subtraction (absolute difference)
        diff = cv2.absdiff(gray, ref_gray)
        
        # 2. Thresholding
        # Frost appears as bright white changes compared to the dark/metallic background.
        # We use a threshold to isolate these changes.
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # 3. Morphological Operations to clean up noise
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Save the automatically generated mask
        out_path = os.path.join(out_dir, f"{os.path.splitext(img_name)[0]}_mask.png")
        cv2.imwrite(out_path, mask)
        
    print(f"Processed {folder_name}, generated masks saved to {out_dir}")

def main():
    base_dir = r"c:\Users\Office\Desktop\NUST MTS\Sir Hassan Elahi Internship\Frost_images-20260709T051618Z-3-001\Frost_images"
    output_dir = r"c:\Users\Office\Desktop\NUST MTS\Sir Hassan Elahi Internship\annotations"
    
    folders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    
    for folder in folders:
        process_folder(folder, output_dir)
        
    print("\n✅ Auto-annotation complete! You can now refine these masks in Photoshop or LabelMe.")

if __name__ == "__main__":
    main()
