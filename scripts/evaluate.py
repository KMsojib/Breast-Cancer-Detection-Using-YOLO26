from ultralytics import YOLO

def main():
    # Load your trained best weights
    model = YOLO("results/weights/best.pt")
    
    # Run evaluation on the test split
    print("Running evaluation on the test split...")
    metrics = model.val(
        data="dataset/data.yaml", 
        split="test",
        imgsz=640,
        batch=16
    )
    
    # Print out core test evaluation metrics
    print("\n--- Test Split Evaluation Results ---")
    print(f"Box mAP50:    {metrics.box.map50:.4f}")
    print(f"Box mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.p.mean():.4f}")
    print(f"Recall:       {metrics.box.r.mean():.4f}")

if __name__ == "__main__":
    main()