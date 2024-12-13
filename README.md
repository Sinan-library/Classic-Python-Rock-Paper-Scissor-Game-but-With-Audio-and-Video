
# Rock Paper Scissors Game (Video & Audio Based)

This project is a Rock, Paper, Scissors game that uses video and audio inputs to determine the player's choice. The game leverages computer vision and speech recognition technologies.

## Features

- **Video Input Mode:** Detects hand gestures using a webcam.
- **Audio Input Mode:** Recognizes spoken words using a microphone.
- **AI Model:** Uses Hugging Face's Transformers for image classification.

## Requirements

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

### Packages Used

- `opencv-python` - For capturing video input.
- `numpy` - For numerical operations.
- `speechrecognition` - For audio input recognition.
- `transformers` - For image classification using a pre-trained model.

## How to Play

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Rock-Paper-Scissors-Game.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Rock-Paper-Scissors-Game
   ```
3. Run the game:
   ```bash
   python rock_paper_scissors_video_audio.py
   ```
4. Follow on-screen instructions to choose video or audio mode.

## How It Works

- In **Video Mode**, the webcam captures an image of your hand gesture and predicts the gesture using the AI model.
- In **Audio Mode**, the microphone captures your spoken word, which is processed using speech recognition.

## Contribution

Feel free to submit issues or pull requests to improve the game.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Happy Playing! 🎮

---

<p align="center"><strong>Developed with ❤️ by Sinan Bandi</strong></p> 