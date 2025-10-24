import cv2
import numpy as np
import base64
import io
import librosa
import streamlit as st
from scipy.stats import beta
import sounddevice as sd
from collections import Counter

# Configurable scale
TARGET_SCALE = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C']
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Decode base64 to video file
def decode_base64_video(base64_clip):
    if base64_clip.startswith('data:video/webm;'):
        base64_clip = base64_clip.split(',')[1]
    video_bytes = base64.b64decode(base64_clip)
    video_path = 'temp.webm'
    with open(video_path, 'wb') as f:
        f.write(video_bytes)
    return video_path

# Extract audio from video
def extract_audio(video_path):
    try:
        y, sr = librosa.load(video_path, sr=22050)
        return y, sr
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None, 22050

# Frequency to note conversion
def freq_to_note(freq, threshold=50.0):
    if freq < threshold:
        return None
    try:
        midi = int(round(69 + 12 * np.log2(freq / 440.0)))
        if midi < 12 or midi > 127:
            return None
        return NOTE_NAMES[midi % 12]
    except (ValueError, OverflowError):
        return None

# Analyze C Major scale
def analyze_scale(y, sr):
    if y is None or len(y) == 0:
        return 0.0, "No audio detected"
    
    y_harmonic = librosa.effects.hpss(y)[0]
    pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), threshold=0.1)
    
    notes = []
    for i in range(pitches.shape[1]):
        if magnitudes[:, i].max() > 0.1:
            index = magnitudes[:, i].argmax()
            freq = pitches[index, i]
            note = freq_to_note(freq)
            if note and magnitudes[index, i] > 0.2:
                notes.append(note)
    
    filtered_notes = []
    current_note, note_count = None, 0
    min_duration = 3
    for note in notes:
        if note == current_note:
            note_count += 1
        else:
            if current_note and note_count >= min_duration:
                filtered_notes.append(current_note)
            current_note = note
            note_count = 1
    if current_note and note_count >= min_duration:
        filtered_notes.append(current_note)
    
    played = filtered_notes[:len(TARGET_SCALE)]
    correct_count = sum(1 for i, note in enumerate(played) if i < len(TARGET_SCALE) and note == TARGET_SCALE[i])
    score = correct_count / len(TARGET_SCALE) if played else 0.0
    a, b = 1 + score * 10, 1 + (1 - score) * 10
    feedback = "Escala Dó Maior correta!" if played == TARGET_SCALE else f"Escala incompleta: {correct_count}/{len(TARGET_SCALE)} notas corretas"
    return beta.rvs(a, b), feedback

# Analyze posture
def analyze_posture(frames):
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    gray = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    posture_score = len(contours) / 100.0 if contours else 0.5
    a, b = 1 + posture_score * 10, 1 + (1 - posture_score) * 10
    return beta.rvs(a, b), "Boa postura" if posture_score > 0.5 else "Melhorar postura"

# Detect guitar with improved template matching
def detect_guitar(frames):
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    gray = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    # Assume guitar has distinct edge patterns
    has_guitar = np.mean(edges) > 20
    a, b = 1 + has_guitar * 10, 1 + (1 - has_guitar) * 10
    return beta.rvs(a, b) > 0.5, "Violão detectado" if has_guitar else "Sem violão"

# Analyze frame
def analyze_frame(frames):
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    gray = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3), np.uint8)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    complexity = np.std(sobel)
    occupation = complexity / 255
    h, w = gray.shape
    center_dist = np.mean([np.sqrt((i - h/2)**2 + (j - w/2)**2) for i in range(h) for j in range(w) if gray[i,j] > 100])
    organized = occupation < 0.5 and center_dist > 100
    a, b = 1 + organized * 10, 1 + (1 - organized) * 10
    return beta.rvs(a, b), "Quadro organizado" if organized else "Quadro confuso"

# Analyze facial expressions
def analyze_facial(frames):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = 0
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3,3), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        detected = face_cascade.detectMultiScale(gray, 1.3, 5)
        faces += len(detected)
    score = faces / len(frames)
    a, b = 1 + score * 10, 1 + (1 - score) * 10
    return beta.rvs(a, b), "Expressões capturadas" if score > 0.5 else "Melhorar expressões"

# Streamlit App
st.title('Plexo Natural Clip Analyzer')

base64_clip = st.text_area('Paste Base64 Clip')
if base64_clip:
    video_path = decode_base64_video(base64_clip)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if frames:
        st.video(video_path)  # Live frame preview
        y, sr = extract_audio(video_path)
        
        report = {
            'scale': analyze_scale(y, sr),
            'posture': analyze_posture(frames),
            'has_guitar': detect_guitar(frames),
            'frame': analyze_frame(frames),
            'facial': analyze_facial(frames)
        }
        
        st.json(report)
    else:
        st.error('No frames in video')
