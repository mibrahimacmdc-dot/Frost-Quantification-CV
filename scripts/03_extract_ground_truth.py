import os
import cv2
import numpy as np
import json
import csv
import glob

# pixels per mm
PIXELS_PER_MM = 3.4

def extract_thickness(mask_roi):
    """
    Given a binary mask of an ROI, finds the maximum horizontal thickness of the frost.
    Since fins are vertical, we look at each row (y) and count the frost pixels (x).
    """
    # Count non-zero (white) pixels in each row
    row_thicknesses = np.sum(mask_roi > 0, axis=1) / 255.0
    
    if len(row_thicknesses) == 0:
        return 0.0
        
    # The max horizontal width in pixels
    max_pixels = np.max(row_thicknesses)
    
    # Convert to mm
    thickness_mm = max_pixels / PIXELS_PER_MM
    return round(thickness_mm, 3)

def main():
    base_dir = r"c:\Users\Office\Desktop\NUST MTS\Sir Hassan Elahi Internship"
    roi_json_path = os.path.join(base_dir, "config", "roi_coordinates.json")
    masks_dir = os.path.join(base_dir, "annotations")
    output_csv = os.path.join(base_dir, "outputs", "ground_truth_thickness.csv")
    
    if not os.path.exists(roi_json_path):
        print("Error: roi_coordinates.json not found. Run 02_roi_definition.py first.")
        return
        
    with open(roi_json_path, 'r') as f:
        roi_data = json.load(f)
    rois = roi_data['rois']
    
    # We will store results as a list of dicts
    results = []
    
    # Find all generated masks
    mask_files = glob.glob(os.path.join(masks_dir, "**", "*_mask.png"), recursive=True)
    
    for mask_path in mask_files:
        folder_name = os.path.basename(os.path.dirname(mask_path))
        img_name = os.path.basename(mask_path).replace("_mask.png", ".png")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        row_data = {
            "Folder": folder_name,
            "Image": img_name
        }
        
        for roi in rois:
            roi_id = roi['id']
            x1, y1, x2, y2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']
            
            # Extract the ROI from the mask
            mask_crop = mask[y1:y2, x1:x2]
            
            # Calculate thickness
            thickness_mm = extract_thickness(mask_crop)
            
            row_data[f"ROI_{roi_id}"] = thickness_mm
            
        results.append(row_data)
        
    if not results:
        print("No masks found to process.")
        return
        
    # Write to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = ["Folder", "Image"] + [f"ROI_{i}" for i in range(1, 21)]
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"✅ Ground truth extraction complete! Saved {len(results)} records to {output_csv}")

if __name__ == "__main__":
    main()
