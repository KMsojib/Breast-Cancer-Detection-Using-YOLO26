import os
import glob
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO, solutions

def generate_predictions_grid(model, test_dir="dataset/images/test", output_path="assets/prediction_3x3_grid.jpg"):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
    
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images for prediction grid.")

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(9):
        ax = axes[i]
        if i < len(image_paths):
            results = model.predict(source=image_paths[i], conf=0.25, verbose=False)
            annotated_img = results[0].plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            ax.imshow(annotated_img)
            ax.set_title(f"Prediction {i+1}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3x3 prediction grid to {output_path}")

def generate_gradcam_grid(model_path, gradcam_dir, output_path="assets/gradcam_heatmap.jpg"):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(gradcam_dir, ext)))
    
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images in Grad-CAM directory.")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(6):
        ax = axes[i]
        if i < len(image_paths):
            img = cv2.imread(image_paths[i])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(f"Grad-CAM Sample {i+1}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 2x3 Grad-CAM grid to {output_path}")

def main():
    model_weight_path = "results/weights/best.pt"
    model = YOLO(model_weight_path)
    
    # Generate Prediction Grid
    generate_predictions_grid(model)
    
    # Generate Grad-CAM Grid from local folder
    gradcam_source_dir = r"C:\Users\kawsa\Desktop\BreastCanncer\Breast Cancer\Grad-Cam view"
    generate_gradcam_grid(model_weight_path, gradcam_source_dir)

if __name__ == "__main__":
    main()