# Music Learn

A Python application that records audio input and analyzes whether a student played a C major scale correctly in melodic sequence.

## Features

-  **Real-time audio recording** using your computer's microphone
-  **Intelligent note detection** using advanced pitch tracking algorithms
-  **Scale analysis** with detailed feedback on accuracy
-  **Configurable scales** - easily change to analyze different scales
-  **Educational feedback** showing exactly which notes were correct/incorrect
-  **Noise filtering** to improve detection accuracy in real-world conditions

## Requirements

### System Requirements
- Python 3.7 or higher
- A working microphone (built-in or external)
- Audio drivers properly installed

### Python Dependencies
```bash
pip install sounddevice librosa numpy
```

## Installation

1. **Clone or download the repository:**
   ```bash
https://github.com/ismaeltrabuco/music-learn)
   ```

2. **Install dependencies:**
  
   ```bash
   pip install sounddevice librosa numpy
   ```

3. **Test your audio setup:**
   ```bash
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```

## Usage

### Basic Usage

1. **Run the application:**
   ```bash
   python music_learn.py
   ```

2. **Follow the on-screen instructions:**
   - The program will count down from 3
   - Play the C major scale (C D E F G A B C) melodically
   - Keep a steady tempo and play one note at a time
   - Wait for analysis results

### Example Output
```
==================================================
Music Learn
==================================================
Instructions:
- Play the C major scale: C D E F G A B C
- Play melodically (one note at a time)
- Keep a steady tempo
- Recording will start in 3 seconds...
==================================================
Starting in 3...
Starting in 2...
Starting in 1...
🎵 Recording started!
Recording for 8 seconds...
🔍 Analyzing notes...
Detected notes: ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C']
Expected scale: ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C']
Total notes detected: 8
✅ Success: C Major scale played correctly!
```

## Customization

### Changing the Target Scale

Edit the `TARGET_SCALE` variable at the top of the file:

```python
# For D Major scale
TARGET_SCALE = ['D', 'E', 'F#', 'G', 'A', 'B', 'C#', 'D']

# For A Minor scale (natural)
TARGET_SCALE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'A']

# For G Major scale
TARGET_SCALE = ['G', 'A', 'B', 'C', 'D', 'E', 'F#', 'G']
```

### Adjusting Recording Settings

Modify the recording parameters in the `record_audio()` function:

```python
def record_audio(duration=8, fs=22050):  # 8 seconds, 22kHz sample rate
```

### Fine-tuning Detection

Adjust detection sensitivity in the `detect_notes()` function:

```python
# Minimum confidence for note detection
if magnitudes[:, i].max() > 0.1:  # Lower = more sensitive

# Minimum duration for a note to be valid
min_duration = 3  # Higher = requires longer notes
```

## Troubleshooting

### Common Issues

**1. "No notes detected"**
- Check your microphone is working and not muted
- Play louder or closer to the microphone
- Ensure your instrument is in tune
- Try playing more slowly with clear separation between notes

**2. "Recording failed"**
- Check audio device permissions
- Try running: `python -c "import sounddevice as sd; print(sd.query_devices())"`
- Make sure no other applications are using the microphone

**3. "Incorrect scale detection"**
- Play more slowly with clear articulation
- Ensure proper tuning of your instrument
- Avoid playing in very high or very low registers
- Check for background noise

**4. ImportError for dependencies:**
```bash
# Update pip first
pip install --upgrade pip

# Install each dependency individually
pip install sounddevice
pip install librosa
pip install numpy
```

### Audio Device Issues

**List available audio devices:**
```python
import sounddevice as sd
print(sd.query_devices())
```

**Set specific input device:**
```python
# In the record_audio function, add device parameter
sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32', device=DEVICE_ID)
```

## Technical Details

### Algorithm Overview

1. **Audio Recording**: Captures 8 seconds of mono audio at 22kHz
2. **Harmonic Separation**: Separates harmonic content from percussive for better pitch detection
3. **Pitch Tracking**: Uses librosa's `piptrack` with frequency range filtering (C2-C7)
4. **Note Conversion**: Converts frequencies to note names using equal temperament tuning
5. **Filtering**: Removes noise, short notes, and consecutive duplicates
6. **Analysis**: Compares detected sequence with target scale

### Frequency Range
- **Minimum**: C2 (65.4 Hz)
- **Maximum**: C7 (2093 Hz)
- **Reference**: A4 = 440 Hz

### Performance Considerations
- Processing time: ~2-3 seconds for 8 seconds of audio
- Memory usage: ~50-100 MB during processing
- CPU usage: Moderate during analysis phase

## Contributing

Feel free to contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License for non-commercial use only.
Non-Commercial Use

- Personal projects and learning
- Educational purposes
- Research and development
- Open source contributions

#### Commercial Use
For commercial use, licensing, or integration into commercial products, please contact:
contato@plexonatural.com

We offer flexible commercial licensing options for businesses and organizations.
## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Test with a simple scale played slowly and clearly
4. Check your audio device configuration
---
**Happy practicing!**
