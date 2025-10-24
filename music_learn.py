import cv2
import numpy as np
import base64
import time
import streamlit as st
import requests
import json

# ----------------------------
# YOLO (optional)
# ----------------------------
try:
    from ultralytics import YOLO
except ImportError:
    st.warning("YOLOv8 not available, using fallback for guitar detection")
    YOLO = None

# ----------------------------
# Pinata upload
# ----------------------------
def upload_to_pinata(file_content, filename, content_type, pinata_jwt):
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    headers = {"Authorization": f"Bearer {pinata_jwt}"}
    files = {'file': (filename, file_content, content_type)}
    response = requests.post(url, headers=headers, files=files)
    if response.status_code == 200:
        return response.json()['IpfsHash']
    else:
        raise Exception(f"Pinata upload failed: {response.text}")

# ----------------------------
# Detectors
# ----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0, faces

def detect_guitar(frame):
    if YOLO:
        try:
            model = YOLO('yolov8n.pt')
            results = model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    if r.names.get(int(box.cls), '') == 'guitar':
                        return True, box.xyxy[0]
        except Exception as e:
            st.warning(f"YOLO error: {e}, using fallback")
    # fallback
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    mean_edges = np.mean(edges)
    if mean_edges > 20:
        return True, [0, 0, frame.shape[1], frame.shape[0]]
    return False, None

# ----------------------------
# Analyses
# ----------------------------
def analyze_posture(faces, frame):
    if len(faces) == 0:
        return False, "No face detected"
    x, y, w, h = faces[0]
    center_x = x + w // 2
    frame_w = frame.shape[1]
    is_good = abs(center_x - frame_w // 2) < frame_w * 0.2
    return is_good, "Boa postura" if is_good else "Melhorar postura: centralize"

def analyze_movement(prev_frame, frame, guitar_box):
    if prev_frame is None or guitar_box is None:
        return "No movement detected"
    diff = cv2.absdiff(prev_frame, frame)
    movement = np.mean(diff) > 10
    return "Movimento detectado" if movement else "Sem movimento"

def analyze_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges) / 255
    return "Quadro organizado" if edge_density < 0.5 else "Quadro confuso"

def analyze_facial(faces):
    return len(faces) > 0, "Expressões capturadas" if len(faces) > 0 else "Melhorar expressões"

# ----------------------------
# Annotations
# ----------------------------
def get_frame_with_annotations(frame, faces, guitar_box, posture_ok):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if guitar_box is not None:
        x1, y1, x2, y2 = map(int, guitar_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        angle = np.arctan2(y2 - y1, x2 - x1 + 1e-5) * 180 / np.pi
        if abs(angle) > 30:
            cv2.putText(frame, "Adjust guitar angle!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if not posture_ok:
        cv2.putText(frame, "Center posture!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

# ----------------------------
# Streamlit App
# ----------------------------
st.title('🎥 Plexo Natural Live Analyzer MVP')
pinata_jwt = st.text_input('Pinata JWT', type='password')
start_button = st.button('Start Live')

if start_button:
    st.info("🔍 Tentando acessar a câmera local...")
    cap = cv2.VideoCapture(0)
    use_local_cam = cap.isOpened()

    if not use_local_cam:
        st.warning("⚠️ Câmera local não disponível — usando câmera do navegador (WebRTC).")
        cap.release()
        frame_placeholder = st.empty()
        frame_input = st.camera_input("📸 Capture live frame")

        if frame_input is None:
            st.stop()

        # Processar apenas 1 frame (modo foto)
        file_bytes = np.asarray(bytearray(frame_input.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        has_face, faces = detect_face(frame)
        has_guitar, guitar_box = detect_guitar(frame)
        posture_ok, posture_feedback = analyze_posture(faces, frame)
        movement = "N/A"
        frame_feedback = analyze_frame(frame)
        facial_score, facial_feedback = analyze_facial(faces)

        annotated_frame = get_frame_with_annotations(frame, faces, guitar_box, posture_ok)
        frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        st.info(f"Face: {'Detected' if has_face else 'Not detected'}")
        st.info(f"Guitar: {'Detected' if has_guitar else 'Not detected'}")
        st.info(f"Posture: {posture_feedback}")
        st.info(f"Movement: {movement}")
        st.info(f"Scenario: {frame_feedback}")

        report = {
            "face_detected": has_face,
            "guitar_detected": has_guitar,
            "good_posture": posture_ok,
            "posture_feedback": posture_feedback,
            "movement": movement,
            "scenario": frame_feedback,
            "facial": facial_feedback
        }
        st.json(report)

        if pinata_jwt:
            try:
                hash = upload_to_pinata(json.dumps(report).encode(),
                                        f'report_{int(time.time())}.json',
                                        'application/json', pinata_jwt)
                st.success(f"✅ JSON uploaded to IPFS: ipfs://{hash}")
            except Exception as e:
                st.error(f"Pinata error: {e}")

    else:
        st.success("✅ Câmera local detectada!")
        frame_placeholder = st.empty()
        prev_frame = None
        frames = []

        for _ in range(150):  # ~5s @ 30fps
            ret, frame = cap.read()
            if not ret:
                st.error('Failed to capture frame')
                break

            frames.append(frame)
            has_face, faces = detect_face(frame)
            has_guitar, guitar_box = detect_guitar(frame)
            posture_ok, posture_feedback = analyze_posture(faces, frame)
            movement = analyze_movement(prev_frame, frame, guitar_box)
            frame_feedback = analyze_frame(frame)
            facial_score, facial_feedback = analyze_facial(faces)

            annotated_frame = get_frame_with_annotations(frame, faces, guitar_box, posture_ok)
            frame_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

            st.info(f"Face: {'Detected' if has_face else 'Not detected'}")
            st.info(f"Guitar: {'Detected' if has_guitar else 'Not detected'}")
            st.info(f"Posture: {posture_feedback}")
            st.info(f"Movement: {movement}")
            st.info(f"Scenario: {frame_feedback}")

            prev_frame = frame
            time.sleep(0.033)

        cap.release()

        # gera vídeo
        if frames:
            height, width = frames[0].shape[:2]
            out = cv2.VideoWriter('temp_clip.webm', cv2.VideoWriter_fourcc(*'vp80'), 30, (width, height))
            for f in frames:
                out.write(f)
            out.release()

            with open('temp_clip.webm', 'rb') as f:
                video_bytes = f.read()
            base64_clip = base64.b64encode(video_bytes).decode()

            report = {
                "face_detected": has_face,
                "guitar_detected": has_guitar,
                "good_posture": posture_ok,
                "posture_feedback": posture_feedback,
                "movement": movement,
                "scenario": frame_feedback,
                "facial": facial_feedback,
                "base64_clip": base64_clip
            }
            st.json(report)

            if pinata_jwt:
                try:
                    hash = upload_to_pinata(json.dumps(report).encode(),
                                            f'report_{int(time.time())}.json',
                                            'application/json', pinata_jwt)
                    st.success(f"✅ JSON uploaded to IPFS: ipfs://{hash}")
                except Exception as e:
                    st.error(f"Pinata error: {e}")
        else:
            st.error("No frames captured")

if __name__ == "__main__":
    st.run()
