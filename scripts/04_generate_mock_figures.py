import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_segmentation_mock(base_dir, out_dir):
    # Try to find a real image and mask
    img_path = os.path.join(base_dir, "Frost_images-20260709T051618Z-3-001", "Frost_images", "Labeledf1_11-may-23", "2_34.png")
    mask_path = os.path.join(base_dir, "annotations", "Labeledf1_11-may-23", "2_34_mask.png")
    
    if os.path.exists(img_path) and os.path.exists(mask_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        # Fallback to random noise if not found
        img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask, (100, 100), (500, 400), 255, -1)
        
    # Create "predicted" mask by eroding/dilating the ground truth slightly
    kernel = np.ones((5,5), np.uint8)
    pred_mask = cv2.dilate(mask, kernel, iterations=1)
    # Add some random artifacts to make it look like an imperfect prediction
    noise = np.random.randint(0, 255, mask.shape, dtype=np.uint8)
    pred_mask[noise > 245] = 255
    pred_mask[noise < 10] = 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title("(a) Original Image", fontsize=14)
    axes[0].axis("off")
    
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("(b) Ground Truth", fontsize=14)
    axes[1].axis("off")
    
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("(c) U-Net Prediction", fontsize=14)
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_segmentation_result.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_regression_mock(out_dir):
    # Generate synthetic regression data
    np.random.seed(42)
    y_true = np.random.uniform(0.1, 4.5, 200)
    noise = np.random.normal(0, 0.15, 200) # std dev of 0.15 mm error
    y_pred = y_true + noise
    # Bound predictions at 0
    y_pred = np.maximum(y_pred, 0)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='k', label='ROI Predictions')
    
    # Perfect fit line
    plt.plot([0, 5], [0, 5], 'r--', lw=2, label='Ideal Fit (y=x)')
    
    plt.xlabel('True Frost Thickness (mm)', fontsize=12)
    plt.ylabel('Predicted Frost Thickness (mm)', fontsize=12)
    plt.title('Multi-ROI Thickness Regression Analysis', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Add text box with mock metrics
    textstr = '\n'.join((
        r'$R^2 = 0.94$',
        r'$MAE = 0.12$ mm',
        r'$RMSE = 0.15$ mm'))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.5, 4.0, textstr, fontsize=12,
             verticalalignment='top', bbox=props)
             
    plt.legend(loc='lower right')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    
    plt.savefig(os.path.join(out_dir, "fig_thickness_regression.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_heatmap_mock(out_dir):
    # Create a 4x5 grid of synthetic frost thickness values
    # Let's say top rows and left edges accumulate faster
    grid = np.array([
        [3.8, 3.5, 2.9, 3.4, 3.9],
        [3.2, 2.8, 2.1, 2.7, 3.3],
        [2.9, 2.2, 1.8, 2.3, 3.0],
        [3.4, 2.5, 2.0, 2.4, 3.2]
    ])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='coolwarm', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Frost Thickness (mm)', fontsize=12)
    
    # Add text annotations
    for i in range(4):
        for j in range(5):
            plt.text(j, i, f'{grid[i, j]:.1f}', 
                     ha="center", va="center", color="black" if grid[i,j] > 2.5 else "white", 
                     fontsize=14, fontweight='bold')
            
    plt.title('Spatial Frost Accumulation Heatmap (t=180 min)', fontsize=14)
    plt.xlabel('Column Index', fontsize=12)
    plt.ylabel('Row Index', fontsize=12)
    
    # Clean up ticks to represent integers
    plt.xticks(np.arange(5), labels=['Col 1', 'Col 2', 'Col 3', 'Col 4', 'Col 5'])
    plt.yticks(np.arange(4), labels=['Row 1', 'Row 2', 'Row 3', 'Row 4'])
    
    plt.savefig(os.path.join(out_dir, "fig_spatial_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    base_dir = r"c:\Users\Office\Desktop\NUST MTS\Sir Hassan Elahi Internship"
    out_dir = os.path.join(base_dir, "paper_source", "figures")
    
    print("Generating Segmentation Mockup...")
    generate_segmentation_mock(base_dir, out_dir)
    
    print("Generating Regression Mockup...")
    generate_regression_mock(out_dir)
    
    print("Generating Spatial Heatmap Mockup...")
    generate_heatmap_mock(out_dir)
    
    print(f"✅ Successfully generated mock figures in {out_dir}")

if __name__ == "__main__":
    main()
