"""
Interactive tool to annotate ball position in video frames for training data.

Usage:
    python -m football_possession.ball_annotation --video path/to/video.mp4 --output path/to/output/dir
    
Controls:
    - SPACE: Mark ball location (click on ball, then press SPACE)
    - N: Skip to next frame
    - P: Go to previous frame
    - S: Save current annotation
    - Q: Quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


class BallAnnotator:
    def __init__(self, video_path: str, output_dir: str):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.current_frame_idx = 0
        self.current_frame = None
        self.display_frame = None
        self.ball_positions: dict[int, tuple[float, float]] = {}  # frame_idx -> (x, y)
        self.current_annotation = None
        
        self._load_annotations()
        self._read_frame()
        
    def _load_annotations(self) -> None:
        """Load any existing annotations from JSON."""
        annotations_file = self.output_dir / "annotations.json"
        if annotations_file.exists():
            with open(annotations_file) as f:
                self.ball_positions = json.load(f)
                self.ball_positions = {int(k): tuple(v) for k, v in self.ball_positions.items()}
    
    def _save_annotations(self) -> None:
        """Save annotations to JSON."""
        annotations_file = self.output_dir / "annotations.json"
        with open(annotations_file, "w") as f:
            json.dump(self.ball_positions, f, indent=2)
        print(f"Saved {len(self.ball_positions)} annotations to {annotations_file}")
    
    def _read_frame(self) -> None:
        """Read current frame and prepare for display."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            print(f"Could not read frame {self.current_frame_idx}")
            return
        
        self.current_frame = frame
        self.display_frame = frame.copy()
        self.current_annotation = self.ball_positions.get(self.current_frame_idx)
        self._draw_display()
    
    def _draw_display(self) -> None:
        """Draw UI elements on display frame."""
        self.display_frame = self.current_frame.copy()
        
        # Draw existing annotation if present
        if self.current_annotation:
            x, y = self.current_annotation
            # Convert from normalized [0, 1] to pixel coordinates if needed
            if x <= 1.0 and y <= 1.0:
                px, py = int(x * self.frame_width), int(y * self.frame_height)
            else:
                px, py = int(x), int(y)
            
            cv2.circle(self.display_frame, (px, py), 8, (0, 255, 0), -1)
            cv2.circle(self.display_frame, (px, py), 10, (0, 255, 0), 2)
        
        # Draw info text
        info = f"Frame {self.current_frame_idx}/{self.total_frames}"
        status = "✓ ANNOTATED" if self.current_annotation else "○ NOT ANNOTATED"
        cv2.putText(self.display_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(self.display_frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (0, 255, 0) if self.current_annotation else (0, 0, 255), 2)
        
        cv2.putText(self.display_frame, "CONTROLS:", (10, self.frame_height - 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(self.display_frame, "CLICK on ball + SPACE = annotate", (10, self.frame_height - 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.display_frame, "N = next  |  P = prev  |  S = save  |  Q = quit", (10, self.frame_height - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _on_mouse(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Normalize coordinates to [0, 1]
            norm_x = x / self.frame_width
            norm_y = y / self.frame_height
            self.current_annotation = (norm_x, norm_y)
            self._draw_display()
            print(f"Ball marked at normalized coords ({norm_x:.3f}, {norm_y:.3f})")
    
    def run(self) -> None:
        """Run the annotation tool."""
        print(f"Opened video: {self.video_path}")
        print(f"Total frames: {self.total_frames}, FPS: {self.fps:.2f}")
        print(f"Output directory: {self.output_dir}")
        print("\nKey controls:")
        print("  SPACE: Save current annotation")
        print("  N: Next frame")
        print("  P: Previous frame")
        print("  S: Save all annotations to file")
        print("  Q: Quit")
        
        cv2.namedWindow("Ball Annotation Tool", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ball Annotation Tool", 1000, 700)
        cv2.setMouseCallback("Ball Annotation Tool", self._on_mouse)
        
        while True:
            cv2.imshow("Ball Annotation Tool", self.display_frame)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                if self.current_frame_idx < self.total_frames - 1:
                    self.current_frame_idx += 1
                    self._read_frame()
                else:
                    print("Already at last frame")
            elif key == ord('p'):
                if self.current_frame_idx > 0:
                    self.current_frame_idx -= 1
                    self._read_frame()
                else:
                    print("Already at first frame")
            elif key == ord(' '):  # SPACE
                if self.current_annotation:
                    self.ball_positions[self.current_frame_idx] = self.current_annotation
                    print(f"Frame {self.current_frame_idx}: Annotation saved")
                    if self.current_frame_idx < self.total_frames - 1:
                        self.current_frame_idx += 1
                        self._read_frame()
                else:
                    print("Click on the ball first!")
            elif key == ord('s'):
                self._save_annotations()
            elif key == ord('d'):  # DELETE annotation
                if self.current_frame_idx in self.ball_positions:
                    del self.ball_positions[self.current_frame_idx]
                    self.current_annotation = None
                    self._draw_display()
                    print(f"Annotation deleted for frame {self.current_frame_idx}")
        
        self._save_annotations()
        cv2.destroyAllWindows()
        self.cap.release()
        print("Annotation tool closed.")


def main():
    parser = argparse.ArgumentParser(description="Annotate ball position in video frames")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", required=True, help="Output directory for annotations")
    
    args = parser.parse_args()
    
    annotator = BallAnnotator(args.video, args.output)
    annotator.run()


if __name__ == "__main__":
    main()
