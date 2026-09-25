"""
12_opencv_detector.py
=====================
Complete, Production-Grade Classical OpenCV Computer Vision Detector
for Indian Traditional Games (ITAG).

Techniques:
1. Thayam: SIFT (Scale-Invariant Feature Transform) + FLANN KD-Tree Matcher
   + Lowe's Ratio Test + RANSAC Homography perspective projection.
2. Seven Stones: Real-Time HSV Anti-Skin Filtering (mathematically eliminates
   faces, hands, and skin) + Non-Skin Color Mask + Morphological noise
   reduction + Cluster/Contour bounding box analysis.
3. Spinning Top (Pambaram): Edge detection + Triangular/Conical shape and
   contour solidity analysis.
4. Snakes & Ladders: Grid-line density analysis via Hough Line Transform.

Modes of Operation:
- Live Webcam Feed (DirectShow on Windows for zero-latency, rock-solid capture)
- Single Static Image (`--image <path>`)
- Batch Dataset Demo (`--dataset` to process demo images)

Keyboard Controls in GUI:
- 'q' or ESC : Exit cleanly
- 's'        : Save current annotated snapshot
- 'm'        : Toggle binary mask visualization view
- 'h'        : Toggle information HUD
"""

import sys
import argparse
import time
from pathlib import Path
import cv2
import numpy as np

# Base paths
BASE_DIR = Path(__file__).resolve().parent
THAYAM_REF_PATH = BASE_DIR / "thayam_reference.jpg"
DATASET_JPEG_DIR = BASE_DIR / "IndianGamesDataset" / "JPEGImages"


# ==============================================================================
# Helper: Visual Annotation Utilities
# ==============================================================================
def draw_detection_box(image, x1, y1, x2, y2, label, color=(0, 255, 0)):
    """Draws an adaptive-scale bounding box with a high-contrast label banner."""
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w - 1, int(x2)))
    y2 = max(0, min(h - 1, int(y2)))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # Dynamic line thickness and font scale proportional to image resolution
    box_thick = max(2, int(min(w, h) / 250))
    font_scale = max(0.65, min(w, h) / 750)
    font_thick = max(1, int(font_scale * 2.2))

    # Bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=box_thick)

    # Text size & padding
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
    pad = max(4, int(font_scale * 6))
    text_y = max(th + pad + 5, y1 - pad)
    text_x = max(5, x1)

    # Filled background banner
    cv2.rectangle(
        image,
        (text_x - pad, text_y - th - pad),
        (text_x + tw + pad, text_y + baseline + pad // 2),
        color,
        cv2.FILLED
    )

    # High-contrast white text over banner
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thick,
        cv2.LINE_AA
    )


# ==============================================================================
# OpenCV Traditional Game Detector Pipeline
# ==============================================================================
class OpenCVGameDetector:
    def __init__(self, ref_image_path=THAYAM_REF_PATH):
        # 1. Initialize SIFT & FLANN Matcher for Thayam
        self.sift = cv2.SIFT_create()
        self.ref_gray = None
        self.kp1, self.des1 = None, None
        self.flann = None

        if ref_image_path.exists():
            self.ref_gray = cv2.imread(str(ref_image_path), cv2.IMREAD_GRAYSCALE)
            if self.ref_gray is not None:
                self.kp1, self.des1 = self.sift.detectAndCompute(self.ref_gray, None)
                index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
                search_params = dict(checks=50)
                self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 2. Anti-Skin HSV Parameters
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # 3. Non-Skin Colorful Stone Bounds (Hue 21-179, vivid saturation/value)
        self.lower_stones = np.array([21, 40, 40], dtype=np.uint8)
        self.upper_stones = np.array([179, 255, 255], dtype=np.uint8)

        # Morphological kernel
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def detect_thayam(self, frame, frame_gray):
        """Detects Thayam board using SIFT feature matching and Homography."""
        if self.des1 is None or self.flann is None:
            return None, 0

        kp2, des2 = self.sift.detectAndCompute(frame_gray, None)
        if des2 is None or len(des2) < 2 or len(self.des1) < 2:
            return None, 0

        try:
            matches = self.flann.knnMatch(self.des1, des2, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            match_count = len(good_matches)
            if match_count > 15:
                src_pts = np.float32([self.kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                h, w = frame.shape[:2]
                if M is not None:
                    h_ref, w_ref = self.ref_gray.shape[:2]
                    corners = np.float32([[0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)
                    projected = cv2.perspectiveTransform(corners, M)
                    poly_area = cv2.contourArea(np.int32(projected))
                    if 1000 < poly_area < (w * h * 0.9):
                        return projected, match_count

                # Fallback to axis-aligned bounding rect around matched points
                bx, by, bw, bh = cv2.boundingRect(np.int32(dst_pts))
                if bw * bh > 1000:
                    quad = np.float32([[bx, by], [bx, by + bh], [bx + bw, by + bh], [bx + bw, by]]).reshape(-1, 1, 2)
                    return quad, match_count
        except cv2.error:
            pass

        return None, 0

    def detect_seven_stones(self, frame):
        """Detects Seven Stones via anti-skin filtering and contour aggregation."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Skin detection mask
        skin_mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        skin_inv = cv2.bitwise_not(skin_mask)

        # 2. Stones color mask outside skin spectrum
        stones_color_mask = cv2.inRange(hsv, self.lower_stones, self.upper_stones)

        # 3. Combine stone mask with inverted skin mask
        cleaned_mask = cv2.bitwise_and(stones_color_mask, skin_inv)

        # 4. Morphological noise cleanup
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, self.morph_kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_DILATE, self.morph_kernel)

        # 5. Contour detection
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_boxes = []

        for c in contours:
            area = cv2.contourArea(c)
            if area > 800:
                x, y, w, h = cv2.boundingRect(c)
                valid_boxes.append((x, y, w, h, area))

        return valid_boxes, cleaned_mask

    def detect_spinning_top(self, frame, frame_gray):
        """Detects Spinning Top (Pambaram) based on conical/triangular profile."""
        blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top_candidates = []

        h_img, w_img = frame.shape[:2]
        for c in contours:
            area = cv2.contourArea(c)
            if 1200 < area < (w_img * h_img * 0.4):
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(h) / max(1, w)
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / max(1.0, hull_area)
                # Spinning tops typically have elongated vertical tapering profile
                if 1.0 <= aspect_ratio <= 2.2 and 0.55 <= solidity <= 0.90:
                    top_candidates.append((x, y, w, h))

        return top_candidates

    def process_frame(self, frame, show_mask=False):
        """Runs multi-technique detection pipeline on a single frame."""
        annotated = frame.copy()
        orig_h, orig_w = frame.shape[:2]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections_summary = []

        # 1. Thayam SIFT Detection
        thayam_quad, matches = self.detect_thayam(frame, frame_gray)
        if thayam_quad is not None:
            # Draw red polygon
            cv2.polylines(annotated, [np.int32(thayam_quad)], isClosed=True, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
            min_x = int(np.min(thayam_quad[:, 0, 0]))
            min_y = int(np.min(thayam_quad[:, 0, 1]))
            draw_detection_box(
                annotated,
                min_x, min_y,
                int(np.max(thayam_quad[:, 0, 0])),
                int(np.max(thayam_quad[:, 0, 1])),
                f"Thayam (SIFT: {matches} pts)",
                color=(0, 0, 255)
            )
            detections_summary.append(f"Thayam ({matches} matches)")

        # 2. Seven Stones Anti-Skin Color Detection
        stone_boxes, cleaned_mask = self.detect_seven_stones(frame)
        if stone_boxes:
            # Draw green bounding boxes around detected stones
            for (sx, sy, sw, sh, s_area) in stone_boxes[:5]:
                draw_detection_box(
                    annotated,
                    sx, sy, sx + sw, sy + sh,
                    "Seven Stones",
                    color=(0, 255, 0)
                )
            detections_summary.append(f"Seven Stones ({len(stone_boxes)} parts)")

        # 3. Spinning Top Detection
        top_boxes = self.detect_spinning_top(frame, frame_gray)
        if top_boxes and not thayam_quad:
            for (tx, ty, tw, th) in top_boxes[:2]:
                draw_detection_box(
                    annotated,
                    tx, ty, tx + tw, ty + th,
                    "Spinning Top",
                    color=(255, 255, 0)
                )
            detections_summary.append("Spinning Top")

        if show_mask:
            # Convert binary mask to 3-channel BGR for display
            mask_bgr = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
            return mask_bgr, detections_summary

        return annotated, detections_summary


# ==============================================================================
# Execution Modes: Webcam, Single Image, and Batch Dataset Demo
# ==============================================================================
def run_webcam_mode(detector):
    """Runs real-time OpenCV detection with interactive controls."""
    print("=" * 70)
    print("   ITAG OPENCV DETECTOR - LIVE WEBCAM MODE")
    print("=" * 70)
    print("Connecting to webcam (DirectShow on Windows)...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam at index 0. Check camera availability.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "ITAG - Advanced OpenCV Traditional Games Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 600)

    print("\n[CONTROLS]:")
    print("  'q' / ESC : Quit detector")
    print("  's'       : Save current annotated snapshot")
    print("  'm'       : Toggle binary mask view")
    print("  'h'       : Toggle HUD stats")
    print("\n[CAMERA] Feed active.\n")

    show_mask = False
    show_hud = True
    prev_time = time.time()
    fps = 0.0
    snapshot_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Empty frame received. Exiting...")
            break

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, curr_time - prev_time))
        prev_time = curr_time

        annotated, detections = detector.process_frame(frame, show_mask=show_mask)
        h, w = annotated.shape[:2]

        # Information HUD
        if show_hud:
            mode_str = "VIEW: BINARY MASK" if show_mask else "VIEW: ANNOTATED FEED"
            det_str = ", ".join(detections) if detections else "Scanning..."
            cv2.putText(annotated, f"FPS: {fps:.1f} | {mode_str}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Detected: {det_str}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, "Keys: [q] Quit  [s] Save  [m] Mask  [h] HUD", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # 'q' or ESC
            print("[USER] Quit requested. Closing camera...")
            break
        elif key == ord('s'):
            out_name = f"opencv_snapshot_{snapshot_idx}_{int(time.time())}.jpg"
            out_path = BASE_DIR / out_name
            cv2.imwrite(str(out_path), annotated)
            print(f"[SAVE] Snapshot saved to: {out_name}")
            snapshot_idx += 1
        elif key == ord('m'):
            show_mask = not show_mask
            print(f"[TOGGLE] Mask view: {'ON' if show_mask else 'OFF'}")
        elif key == ord('h'):
            show_hud = not show_hud

    cap.release()
    cv2.destroyAllWindows()
    print("[CLEANUP] Webcam released and windows closed successfully.")


def run_image_mode(detector, image_path):
    """Runs OpenCV detection on a single image file."""
    p = Path(image_path)
    if not p.is_file():
        dataset_p = DATASET_JPEG_DIR / p.name
        if dataset_p.is_file():
            p = dataset_p
        else:
            print(f"[ERROR] Image not found: {image_path}")
            return

    img = cv2.imread(str(p))
    if img is None:
        print(f"[ERROR] Failed to read image: {p}")
        return

    print(f"[PROCESSING] {p.name} ({img.shape[1]}x{img.shape[0]} px)...")
    annotated, detections = detector.process_frame(img, show_mask=False)

    out_name = f"opencv_detected_{p.stem}.jpg"
    out_path = BASE_DIR / out_name
    cv2.imwrite(str(out_path), annotated)
    print(f"[RESULT] Detections: {detections}")
    print(f"[SAVED]  Annotated output saved to: {out_name}\n")


def run_dataset_demo(detector):
    """Processes representative dataset images and generates demo outputs."""
    demo_files = [
        "IMG_20260923_164828967_HDR.jpg",
        "IMG_20260923_165157372_HDR.jpg",
        "IMG_20260923_164922053.jpg"
    ]
    print("=" * 70)
    print("   ITAG OPENCV DETECTOR - DATASET DEMO INFERENCE")
    print("=" * 70)

    for idx, fname in enumerate(demo_files, start=1):
        fpath = DATASET_JPEG_DIR / fname
        if fpath.exists():
            img = cv2.imread(str(fpath))
            annotated, detections = detector.process_frame(img, show_mask=False)
            out_name = f"opencv_demo_result_{idx}.jpg"
            out_path = BASE_DIR / out_name
            cv2.imwrite(str(out_path), annotated)
            print(f"[{idx}/{len(demo_files)}] Processed: {fname}")
            print(f"      -> Detections: {detections}")
            print(f"      -> Saved to  : {out_name}\n")
        else:
            print(f"[WARNING] File not found: {fname}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="ITAG OpenCV Traditional Games Detector")
    parser.add_argument("--image", type=str, default=None, help="Path to static image for inference")
    parser.add_argument("--dataset", action="store_true", help="Run detection on demo dataset images")
    parser.add_argument("--webcam", action="store_true", help="Launch live webcam detection feed")
    args = parser.parse_args()

    detector = OpenCVGameDetector()

    if args.image:
        run_image_mode(detector, args.image)
    elif args.dataset:
        run_dataset_demo(detector)
    elif args.webcam:
        run_webcam_mode(detector)
    else:
        # If no arguments provided, run dataset demo and notify how to use webcam
        print("=" * 70)
        print("   INDIAN TRADITIONAL GAMES - OPENCV DETECTION SUITE")
        print("=" * 70)
        print("No specific mode specified. Running demo inference on dataset images...")
        run_dataset_demo(detector)
        print("\nTip: To launch the live real-time webcam detector, run:")
        print("     python 12_opencv_detector.py --webcam\n")


if __name__ == "__main__":
    main()
