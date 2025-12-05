import os
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__, template_folder="templates")

# ---------- CONFIG (Matches your provided script) ----------
MODEL_PATH = r"./bestbest.pt"      # Change if needed
CAM_INDEX  = 0                     # Webcam index
FRAME_W    = 1280
FRAME_H    = 720
IMGSZ      = 540
WARNING_PX = 150                   # Distance to trigger WARNING
# -----------------------------------------------------------

# Load model
model = YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
if model is None:
    print(f"WARNING: Model not found at {MODEL_PATH}")

# --- EXACT HELPER FUNCTIONS FROM YOUR SCRIPT ---
def get_box_coords(frame_w, frame_h, box_w=400, box_h=400):
    # Reduced box size to represent a specific object
    cx, cy = frame_w // 2, frame_h // 2
    x1 = cx - box_w // 2
    y1 = cy - box_h // 2
    x2 = cx + box_w // 2
    y2 = cy + box_h // 2
    return int(x1), int(y1), int(x2), int(y2)

def rects_intersect(rectA, rectB):
    ax1, ay1, ax2, ay2 = rectA
    bx1, by1, bx2, by2 = rectB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = inter_x2 - inter_x1
    ih = inter_y2 - inter_y1
    return (iw > 0) and (ih > 0)

def rect_distance(rectA, rectB):
    # Shortest distance between two non-overlapping rectangles
    ax1, ay1, ax2, ay2 = rectA
    bx1, by1, bx2, by2 = rectB
    dx = 0
    if ax2 < bx1:      dx = bx1 - ax2
    elif bx2 < ax1:    dx = ax1 - bx2
    dy = 0
    if ay2 < by1:      dy = by1 - ay2
    elif by2 < ay1:    dy = ay1 - by2
    return int(np.hypot(dx, dy))

def generate_frames():
    """
    This function runs the EXACT loop from your script, 
    but yields JPEG bytes instead of using cv2.imshow
    """
    # Open webcam
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: camera index {CAM_INDEX} not available.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    prev = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        
        # --- LOGIC START ---
        
        # This box represents the "Virtual Object" (Danger Zone)
        x1, y1, x2, y2 = get_box_coords(w, h, box_w=300, box_h=300)

        # Draw the Virtual Object (White box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 2)
        cv2.putText(frame, "DANGER ZONE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

        # YOLO inference
        hand_bbox = None
        if model is not None:
            # verbose=False prevents cluttering the console
            results = model.predict(frame, imgsz=IMGSZ, conf=0.25, max_det=1, verbose=False)
            if len(results) > 0 and len(results[0].boxes) > 0:
                b = results[0].boxes[0]
                xyxy = b.xyxy.cpu().numpy().astype(int)[0]
                hand_bbox = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                
        # --------------------------
        # STATE LOGIC (COPIED EXACTLY)
        # --------------------------
        state = "NO HAND"
        color = (100, 100, 100)
        dist = 0

        if hand_bbox is not None:
            # 1. Check if Hand is TOUCHING/INSIDE the object
            # Note: rects_intersect expects tuple/list, ensure format is correct
            danger_rect = (x1, y1, x2, y2)
            
            is_touching = rects_intersect(hand_bbox, danger_rect)

            if is_touching:
                state = "DANGER"
                color = (0, 0, 255) # Red
                dist = 0
            else:
                # 2. Check distance
                dist = rect_distance(hand_bbox, danger_rect)
                
                if dist <= WARNING_PX:
                    state = "WARNING"
                    color = (0, 165, 255) # Orange
                else:
                    state = "SAFE"
                    color = (0, 255, 0) # Green

            # Draw hand box with state color
            cv2.rectangle(frame, (hand_bbox[0], hand_bbox[1]), (hand_bbox[2], hand_bbox[3]), color, 2)
            
            # Display Stats
            cv2.putText(frame, f"Dist: {dist}px", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # UI Display
        cv2.putText(frame, state, (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)
        
        if state == "DANGER":
            cv2.putText(frame, "HAND TOO CLOSE!", (w//2 - 200, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 4)

        # Show FPS
        now = time.time()
        # Simple moving average for FPS
        fps = 0.9 * fps + 0.1 * (1.0 / (now - prev)) if (now - prev) > 0 else 0.0
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # --- LOGIC END ---

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # debug=False is recommended for cv2 threads
    app.run(host='0.0.0.0', port=5000, debug=False)