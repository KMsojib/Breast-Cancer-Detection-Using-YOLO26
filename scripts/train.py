from ultralytics import YOLO

def main():
    # Load a pre-trained YOLO model (e.g., yolo26n or your specific weights)
    model = YOLO("yolo26n.pt")
    
    # Start training on your custom dataset
    results = model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="breast_cancer_run"
    )
    
    print("Training completed successfully. Best weights saved to results/breast_cancer_run/weights/best.pt")

if __name__ == "__main__":
    main()