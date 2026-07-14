import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def define_rois(ref_image_path):
    img = cv2.imread(ref_image_path)
    if img is None:
        raise ValueError(f"Could not load image at {ref_image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # The horizontal coils are bright metallic.
    # We can find them by taking the horizontal projection (summing pixels across each row)
    # Applying a slight Gaussian blur helps smooth the projection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # We want to find the horizontal coils. We can use a Sobel filter to find horizontal edges,
    # or just use intensity since the coils are bright.
    # Let's use intensity projection
    horizontal_projection = np.sum(blurred, axis=1)
    
    # Find peaks in the horizontal projection
    # The peaks correspond to the thick horizontal coils.
    # We expect 5 major horizontal coils based on the reference image (which form 4 rows of fins)
    from scipy.signal import find_peaks
    # Distance of at least 50 pixels between coils
    peaks, properties = find_peaks(horizontal_projection, distance=50, prominence=10000)
    
    if len(peaks) < 5:
        print(f"Warning: Expected 5 horizontal coils, found {len(peaks)}. Using fallback uniform spacing.")
        # Fallback if peak detection fails
        h = img.shape[0]
        # Ignore top and bottom margins (ruler)
        top_margin = int(h * 0.1)
        bottom_margin = int(h * 0.8) # Ruler is usually at the bottom 20%
        peaks = np.linspace(top_margin, bottom_margin, 5, dtype=int)
    
    # We'll take the top 5 peaks
    peaks = sorted(peaks[:5])
    
    # Define columns (X coordinates)
    # We need 5 columns per row, making 20 ROIs.
    # Exclude the far left and far right edges slightly.
    w = img.shape[1]
    left_edge = int(w * 0.05)
    right_edge = int(w * 0.95)
    col_width = (right_edge - left_edge) // 5
    
    rois = []
    roi_id = 1
    
    # Draw image for visualization
    vis_img = img.copy()
    
    # Create the 20 ROIs
    for i in range(len(peaks) - 1):
        y1 = peaks[i] + 15  # Add a small margin to skip the horizontal coil itself
        y2 = peaks[i+1] - 15
        
        for j in range(5):
            x1 = left_edge + j * col_width
            x2 = left_edge + (j + 1) * col_width
            
            roi = {
                "id": roi_id,
                "row": i + 1,
                "col": j + 1,
                "x1": x1,
                "y1": int(y1),
                "x2": x2,
                "y2": int(y2)
            }
            rois.append(roi)
            
            # Draw rectangle on visualization image
            cv2.rectangle(vis_img, (x1, int(y1)), (x2, int(y2)), (0, 0, 255), 2)
            cv2.putText(vis_img, str(roi_id), (x1+10, int(y1)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            roi_id += 1

    # Save to JSON
    config_dir = r"c:\Users\Office\Desktop\NUST MTS\Sir Hassan Elahi Internship\config"
    os.makedirs(config_dir, exist_ok=True)
    
    json_path = os.path.join(config_dir, "roi_coordinates.json")
    with open(json_path, 'w') as f:
        json.dump({"rois": rois}, f, indent=4)
        
    print(f"✅ Successfully defined 20 ROIs and saved to {json_path}")
    
    # Save visualization
    vis_path = os.path.join(config_dir, "roi_visualization.png")
    cv2.imwrite(vis_path, vis_img)
    print(f"✅ Saved ROI visualization to {vis_path}")

if __name__ == "__main__":
    # Test with one of the standard reference images
    ref_path = r"c:\Users\Office\Desktop\NUST MTS\Sir Hassan Elahi Internship\Frost_images-20260709T051618Z-3-001\Frost_images\Labeledf1_11-may-23\0.png"
    # Need scipy for find_peaks
    try:
        import scipy
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "scipy"])
        
    define_rois(ref_path)
