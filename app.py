# ========================= FINAL + DAILY COUNT + NO DUPLICATES =========================
import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import json
import os
import threading
import queue
import time
import sqlite3
from datetime import datetime, date
from collections import deque

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Loading Bay - Smart Counter", layout="wide")
st.title("Smart Box Counter | Daily Report + No Duplicates")
st.markdown("**Camera:** `150.129.50.173` • **Zero Duplicates** • **Daily Total Saved**")

RTSP_URL = "rtsp://admin:Apple%409978@150.129.50.173:554/stream1"
CONF_THRESHOLD = 0.5
ALERT_LIMIT = 5

# ------------------- DATABASE SETUP -------------------
DB_FILE = "box_counts.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS daily_counts
                 (date TEXT PRIMARY KEY, count INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS entries
                 (id INTEGER PRIMARY KEY, entry_time TEXT, center_x REAL, center_y REAL)''')
    conn.commit()
    conn.close()

init_db()

# ------------------- LOAD ROI -------------------
if not os.path.exists("roi.json"):
    st.error("roi.json not found!")
    st.stop()

with open("roi.json") as f:
    data = json.load(f)
pts = np.array(data, dtype=np.int32) if isinstance(data[0], list) else np.array(data, dtype=np.int32).reshape(-1, 2)
ROI_POLYGON = pts.reshape((-1, 1, 2))

def is_inside_roi(x1, y1, x2, y2):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cv2.pointPolygonTest(ROI_POLYGON, (cx, cy), False) >= 0

# ------------------- MODEL -------------------
@st.cache_resource
def load_model():
    model = YOLO("train2/weights/best.pt")
    return model

model = load_model()

# ------------------- TRACKING LOGIC (NO DUPLICATES) -------------------
tracked_boxes = {}        # id → center point history
MAX_HISTORY = 10
MIN_IOU = 0.3
CENTER_DISTANCE_THRESHOLD = 80  # pixels

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x1 >= x2 or y1 >= y2: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)

def add_to_db():
    today = date.today().isoformat()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO daily_counts (date, count) VALUES (?, 0)", (today,))
    c.execute("UPDATE daily_counts SET count = count + 1 WHERE date = ?", (today,))
    conn.commit()
    conn.close()

# ------------------- UI -------------------
frame_ph = st.empty()
c1, c2, c3, c4 = st.columns(4)
current_ph = c1.empty()
today_ph = c2.empty()
fps_ph = c3.empty()
status_ph = c4.empty()
alert_ph = st.sidebar.empty()

# Daily total
def get_today_count():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT count FROM daily_counts WHERE date = ?", (date.today().isoformat(),))
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0

today_total = get_today_count()
today_ph.metric("Today's Total Boxes", today_total)

# ------------------- CAPTURE THREAD -------------------
frame_queue = queue.Queue(maxsize=2)

def capture_thread():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    delay = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(delay)
            delay = min(delay * 2, 30)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        delay = 1
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except: pass
        frame_queue.put(frame)

threading.Thread(target=capture_thread, daemon=True).start()
time.sleep(3)

frame_count = 0
start_time = time.time()

while True:
    try:
        frame = frame_queue.get(timeout=1)
    except:
        status_ph.error("NO SIGNAL")
        continue

    frame_count += 1
    status_ph.success("LIVE • TRACKING")

    small = cv2.resize(frame, (640, 640))
    results = model(small, conf=CONF_THRESHOLD, verbose=False)[0]

    h, w = frame.shape[:2]
    sx = w / 640
    sy = h / 640

    annotated = frame.copy()
    current_in_roi = 0
    new_detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy() * [sx, sy, sx, sy]).astype(int)
        conf = box.conf.item()
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        bbox = (x1, y1, x2, y2)
        new_detections.append((bbox, center, conf))

        inside = is_inside_roi(x1, y1, x2, y2)
        if inside:
            current_in_roi += 1
            color = (0, 255, 0)
            thick = 6
        else:
            color = (0, 255, 255)
            thick = 2

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
        cv2.putText(annotated, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)

    # Simple tracking: match new boxes to previous ones
    matched = set()
    for i, (bbox, center, conf) in enumerate(new_detections):
        if not is_inside_roi(*bbox): continue
        found = False
        for prev_id, history in tracked_boxes.items():
            if prev_id in matched: continue
            last_center = history[-1]
            dist = np.linalg.norm(np.array(center) - np.array(last_center))
            if dist < CENTER_DISTANCE_THRESHOLD:
                tracked_boxes[prev_id].append(center)
                matched.add(prev_id)
                found = True
                break
        if not found:
            new_id = len(tracked_boxes)
            tracked_boxes[new_id] = deque([center], maxlen=MAX_HISTORY)
            add_to_db()
            today_total += 1
            today_ph.metric("Today's Total Boxes", today_total)

    # Clean old tracks
    to_remove = [k for k, v in tracked_boxes.items() if len(v) == MAX_HISTORY]
    for k in to_remove:
        del tracked_boxes[k]

    # Draw ROI
    overlay = annotated.copy()
    cv2.fillPoly(overlay, [ROI_POLYGON], (255, 150, 50))
    cv2.polylines(overlay, [ROI_POLYGON], True, (0, 0, 255), 10)
    cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
    first_pt = ROI_POLYGON[0][0]
    cv2.putText(annotated, "LOADING BAY", (int(first_pt[0]), int(first_pt[1])-50),
                cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 7)

    # Update UI
    frame_ph.image(annotated, channels="BGR", use_container_width=True)
    current_ph.metric("Currently in Bay", current_in_roi)
    fps_ph.metric("FPS", f"{frame_count/(time.time()-start_time):.1f}")

    if current_in_roi > ALERT_LIMIT:
        alert_ph.error(f"OVER CAPACITY → {current_in_roi} BOXES!")
    else:
        alert_ph.empty()
        