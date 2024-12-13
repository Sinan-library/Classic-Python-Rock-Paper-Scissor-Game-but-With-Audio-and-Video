import cv2
import random
import numpy as np
import speech_recognition as sr
from transformers import pipeline

recognizer = sr.Recognizer()
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

options = ["rock", "paper", "scissors"]

def get_video_choice():
    cap = cv2.VideoCapture(0)
    print("Show your hand gesture to the camera...")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to capture image.")
        return None

    cv2.imwrite("gesture.jpg", frame)
    result = classifier("gesture.jpg")
    label = result[0]['label'].lower()
    if label in options:
        return label
    else:
        print(f"Detected: {label}. Try again.")
        return None

def get_audio_choice():
    with sr.Microphone() as source:
        print("Say Rock, Paper, or Scissors:")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio).lower()
            if text in options:
                return text
            else:
                print(f"Heard: {text}. Try again.")
                return None
        except sr.UnknownValueError:
            print("Sorry, could not understand.")
        except sr.RequestError:
            print("Speech service unavailable.")
        return None

def play_game():
    print("Choose mode: 1 for Video, 2 for Audio")
    mode = input("Enter 1 or 2: ")

    if mode == "1":
        user_choice = get_video_choice()
    elif mode == "2":
        user_choice = get_audio_choice()
    else:
        print("Invalid mode selected.")
        return

    if not user_choice:
        print("No valid choice detected. Game over.")
        return

    computer_choice = random.choice(options)
    print(f"You chose: {user_choice}, Computer chose: {computer_choice}")

    if user_choice == computer_choice:
        print("It's a tie!")
    elif (user_choice == "rock" and computer_choice == "scissors") or \
         (user_choice == "paper" and computer_choice == "rock") or \
         (user_choice == "scissors" and computer_choice == "paper"):
        print("You win!")
    else:
        print("You lose!")

if __name__ == "__main__":
    play_game()
