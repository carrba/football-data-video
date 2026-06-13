from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

# Click in this exact order
TARGETS: list[tuple[str, tuple[float, float]]] = [
    ("goalpost_near", (120.0, 36.0)),
    ("goalpost_far", (120.0, 44.0)),
    ("six_yard_corner_near", (114.0, 30.0)),
    ("six_yard_corner_far", (114.0, 50.0)),
    ("eighteen_yard_corner_near", (102.0, 18.0)),
    ("eighteen_yard_corner_far", (102.0, 63.0)),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Click known pitch landmarks to build a calibration JSON for shot analysis."
    )
    parser.add_argument("--image", required=True, help="Path to shot image")
    parser.add_argument(
        "--output",
        default="calibration_points.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    display = frame.copy()
    clicks: list[tuple[float, float]] = []

    print("Click points in this order:")
    for idx, (name, std) in enumerate(TARGETS, start=1):
        print(f"  {idx}. {name} -> standard {std}")
    print("Press ESC to abort.")

    current_idx = 0

    def _redraw() -> None:
        nonlocal display
        display = frame.copy()
        for i, (px, py) in enumerate(clicks):
            cv2.circle(display, (int(px), int(py)), 6, (0, 255, 255), -1)
            cv2.putText(
                display,
                str(i + 1),
                (int(px) + 8, int(py) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        prompt = (
            "Done" if current_idx >= len(TARGETS) else f"Click {current_idx+1}/{len(TARGETS)}: {TARGETS[current_idx][0]}"
        )
        cv2.putText(
            display,
            prompt,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def on_mouse(event, x, y, _flags, _param):
        nonlocal current_idx
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if current_idx >= len(TARGETS):
            return
        clicks.append((float(x), float(y)))
        current_idx += 1
        _redraw()

    window = "Landmark Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    _redraw()

    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            print("Cancelled.")
            return
        if current_idx >= len(TARGETS):
            break

    cv2.destroyAllWindows()

    payload = []
    for (name, standard), pixel in zip(TARGETS, clicks, strict=True):
        payload.append(
            {
                "name": name,
                "pixel": [pixel[0], pixel[1]],
                "standard": [standard[0], standard[1]],
            }
        )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved calibration landmarks to: {output_path}")


if __name__ == "__main__":
    main()
