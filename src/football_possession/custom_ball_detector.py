"""
Integration utilities for using custom-trained ball detector models.

This module provides utilities to easily switch between pre-trained and custom models
in your detection pipeline.
"""

from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO


class CustomBallDetector:
    """Wrapper for using a custom-trained YOLO ball detection model."""
    
    def __init__(self, model_path: str):
        """
        Initialize with a custom trained model.
        
        Args:
            model_path: Path to trained model (usually ends with .pt)
                       Example: "runs/detect/ball_detector/weights/best.pt"
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(str(self.model_path))
        print(f"Loaded custom ball detector: {model_path}")
    
    def detect(self, frame, confidence: float = 0.5) -> list[tuple[tuple[float, float, float, float], float]]:
        """
        Detect ball in frame.
        
        Returns:
            List of (bbox, confidence) tuples
            bbox = (x1, y1, x2, y2) in pixel coordinates
        """
        results = self.model.predict(source=frame, conf=confidence, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            detections.append(((x1, y1, x2, y2), conf))
        
        return detections


# Example usage
if __name__ == "__main__":
    import cv2
    
    # Load custom model
    detector = CustomBallDetector("runs/detect/ball_detector/weights/best.pt")
    
    # Load image
    frame = cv2.imread("test_frame.jpg")
    if frame is None:
        print("No test image found. Skipping demo.")
    else:
        # Detect ball
        detections = detector.detect(frame, confidence=0.5)
        
        print(f"Found {len(detections)} ball detections")
        for bbox, conf in detections:
            x1, y1, x2, y2 = bbox
            print(f"  BBox: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), Confidence: {conf:.3f}")
            
            # Draw on image
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite("output_with_detections.jpg", frame)
        print("Saved annotated image: output_with_detections.jpg")
