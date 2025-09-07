import sounddevice as sd
import numpy as np
import librosa
import time
from collections import Counter

# Program Description: Records audio input, detects notes, and analyzes if the student played a C major scale (C D E F G A B C) melodically in 4/4 time.

# --- CONFIGURABLE SCALE ---
# Change this list to analyze a different scale (e.g., ['D', 'E', 'F#', 'G', 'A', 'B', 'C#', 'D'] for D major)
TARGET_SCALE = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C']

# --- NOTE FREQUENCY MAPPING ---
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_note(freq, threshold=50.0):
    """
    Convert frequency to note name with minimum frequency threshold
    """
    if freq < threshold:  # Ignore very low frequencies (likely noise)
        return None
    
    try:
        # MIDI note number calculation
        midi = int(round(69 + 12 * np.log2(freq / 440.0)))
        
        # Ensure MIDI is in reasonable range (C0 to B8)
        if midi < 12 or midi > 127:
            return None
            
        note = NOTE_NAMES[midi % 12]
        return note
    except (ValueError, OverflowError):
        return None

def record_audio(duration=8, fs=22050):
    """
    Record audio with error handling
    """
    print(f"Recording for {duration} seconds...")
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        return audio.flatten(), fs
    except Exception as e:
        print(f"Error during recording: {e}")
        return None, fs

def detect_notes(y, sr, hop_length=512, frame_length=2048):
    """
    Detect notes with improved filtering and minimum duration
    """
    if y is None or len(y) == 0:
        return []
    
    # Use harmonic component for better pitch detection
    y_harmonic = librosa.effects.hpss(y)[0]
    
    # Detect pitches
    pitches, magnitudes = librosa.piptrack(
        y=y_harmonic, 
        sr=sr, 
        hop_length=hop_length,
        fmin=librosa.note_to_hz('C2'),  # Minimum frequency (C2)
        fmax=librosa.note_to_hz('C7'),  # Maximum frequency (C7)
        threshold=0.1
    )
    
    notes_with_confidence = []
    
    for i in range(pitches.shape[1]):
        # Find the strongest pitch in this frame
        if magnitudes[:, i].max() > 0.1:  # Minimum confidence threshold
            index = magnitudes[:, i].argmax()
            freq = pitches[index, i]
            confidence = magnitudes[index, i]
            
            note = freq_to_note(freq)
            if note and confidence > 0.2:  # Additional confidence filter
                notes_with_confidence.append((note, confidence))
    
    # Extract just the notes
    notes = [note for note, _ in notes_with_confidence]
    
    # Remove consecutive duplicates and filter short notes
    filtered_notes = []
    current_note = None
    note_count = 0
    min_duration = 3  # Minimum frames for a note to be considered valid
    
    for note in notes:
        if note == current_note:
            note_count += 1
        else:
            if current_note and note_count >= min_duration:
                filtered_notes.append(current_note)
            current_note = note
            note_count = 1
    
    # Don't forget the last note
    if current_note and note_count >= min_duration:
        filtered_notes.append(current_note)
    
    return filtered_notes

def analyze_scale(played_notes, target_scale):
    """
    Analyze if the played notes match the target scale
    """
    if not played_notes:
        print("No notes detected. Please try again with clearer playing.")
        return
    
    # Only consider the first len(target_scale) notes
    played = played_notes[:len(target_scale)]
    
    print(f"Detected notes: {played}")
    print(f"Expected scale: {target_scale}")
    print(f"Total notes detected: {len(played_notes)}")
    
    if len(played) < len(target_scale):
        print(f"Incomplete scale: Only {len(played)} out of {len(target_scale)} notes detected.")
        print("Try playing more clearly or for a longer duration.")
        return
    
    # Check exact match
    if played == target_scale:
        print("✅ Success: C Major scale played correctly!")
    else:
        print("❌ Try again: The scale was not played correctly.")
        
        # Provide detailed feedback
        correct_count = sum(1 for i, note in enumerate(played) if i < len(target_scale) and note == target_scale[i])
        print(f"Accuracy: {correct_count}/{len(target_scale)} notes correct")
        
        # Show differences
        for i, (expected, actual) in enumerate(zip(target_scale, played)):
            status = "✅" if expected == actual else "❌"
            print(f"  Position {i+1}: Expected {expected}, Got {actual} {status}")

def main():
    """
    Main function with improved user interaction
    """
    print("=" * 50)
    print("C MAJOR SCALE ANALYZER")
    print("=" * 50)
    print("Instructions:")
    print("- Play the C major scale: C D E F G A B C")
    print("- Play melodically (one note at a time)")
    print("- Keep a steady tempo")
    print("- Recording will start in 3 seconds...")
    print("=" * 50)
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print("🎵 Recording started!")
    audio, fs = record_audio()
    
    if audio is not None:
        print("🔍 Analyzing notes...")
        notes = detect_notes(audio, fs)
        analyze_scale(notes, TARGET_SCALE)
    else:
        print("❌ Recording failed. Please check your audio device.")
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()
