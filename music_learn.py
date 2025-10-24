import cv2
import numpy as np
import base64
import time
import streamlit as st
import requests
from ultralytics import YOLO  # YOLO8 for guitar detection
import json

# Pinata upload
def upload_to_pinata(file_content, filename, content_type, pinata_jwt):
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    headers = {"Authorization": f"Bearer {pinata_jwt}"}
    files = {'file': (filename, file_content, content_type)}
    response = requests.post(url, headers=headers, files=files)
    if response.status_code == 200:
        return response.json()['IpfsHash']
    else:
        raise Exception(f"Pinata upload failed: {response.text}")

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
try:
    guitar_model = YOLO('yolov8n.pt')  # Load YOLO8
except:
    st.warning("YOLO8 not available, using fallback for guitar detection")

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0, faces

def detect_guitar(frame):
    try:
        results = guitar_model(frame)
        for r in results:
            for box in r.boxes:
                if r.names[int(box.cls)] == 'guitar':  # Adjust class ID if needed
                    return True, box.xyxy[0]
        return False, None
    except:
        # Fallback: edge density
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        mean_edges = np.mean(edges)
        if mean_edges > 20:
            return True, [0, 0, frame.shape[1], frame.shape[0]]  # Dummy box
        return False, None

def analyze_posture(faces):
    if len(faces) == 0:
        return False
    x, y, w, h = faces[0]
    center_x = x + w // 2
    frame_w = frame.shape[1]
    is_good = abs(center_x - frame_w // 2) < frame_w * 0.2
    return is_good

def analyze_movement(prev_frame, frame, guitar_box):
    if prev_frame is None or guitar_box is None:
        return "No movement detected"
    diff = cv2.absdiff(prev_frame, frame)
    movement = np.mean(diff) > 10
    return "Movement detected" if movement else "No movement"

def analyze_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges) / 255
    return "Quadro organizado" if edge_density < 0.5 else "Quadro confuso"

def analyze_facial(faces):
    return len(faces) > 0, "Expressões capturadas" if len(faces) > 0 else "Melhorar expressões"

def get_frame_with_annotations(frame, faces, guitar_box):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Face frame
    if guitar_box:
        x1, y1, x2, y2 = map(int, guitar_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Guitar frame
        # Posture line (dummy)
        cv2.line(frame, (frame.shape[1]//2, 0), (frame.shape[1]//2, frame.shape[0]), (0, 0, 255), 1)
    return frame

st.title('Live Analyzer MVP')
pinata_jwt = st.text_input('Pinata JWT', type='password')
start_button = st.button('Start Live')

if start_button:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error('Error opening camera. Check device.')
        st.stop()
    
    frame_placeholder = st.empty()
    prev_frame = None
    frames = []  # Save for base64
    
    for _ in range(150):  # 5s @ 30fps
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        has_face, faces = detect_face(frame)
        has_guitar, guitar_box = detect_guitar(frame)
        good_posture = analyze_posture(faces)
        movement = analyze_movement(prev_frame, frame, guitar_box)
        frame_feedback = analyze_frame(frame)
        facial_score, facial_feedback = analyze_facial(faces)
        
        annotated_frame = get_frame_with_annotations(frame, faces, guitar_box)
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        
        if not good_posture:
            st.warning("Alert: Improve posture!")
        if has_guitar:
            x1, y1, x2, y2 = map(int, guitar_box)
            angle = np.arctan((y2 - y1) / (x2 - x1 + 1e-5)) * 180 / np.pi
            if abs(angle) > 30:
                st.warning("Alert: Adjust guitar angle!")
        st.info(f"Scenario: {frame_feedback}")
        
        prev_frame = frame
        time.sleep(0.033)
    
    cap.release()
    
    # Generate base64 clip
    height, width = frames[0].shape[:2]
    video_writer = cv2.VideoWriter('temp_clip.webm', cv2.VideoWriter_fourcc(*'vp80'), 30, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()
    
    with open('temp_clip.webm', 'rb') as f:
        video_bytes = f.read()
    base64_clip = base64.b64encode(video_bytes).decode()
    
    # Generate JSON feedback
    report = {
        "face_detected": has_face,
        "guitar_detected": has_guitar,
        "good_posture": good_posture,
        "movement": movement,
        "scenario": frame_feedback,
        "facial": facial_feedback
    }
    st.json(report)
    
    if pinata_jwt:
        json_buffer = json.dumps(report).encode()
        try:
            hash = upload_to_pinata(json_buffer, 'report.json', 'application/json', pinata_jwt)
            st.success(f"JSON uploaded to IPFS: {hash}")
        except Exception as e:
            st.error(f"Pinata error: {e}")

if __name__ == "__main__":
    st.run()
