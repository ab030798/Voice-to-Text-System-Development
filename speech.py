import speech_recognition as sr
import librosa
import numpy as np
import queue
import sounddevice as sd
import torch
import torchaudio


def preprocess_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=16000) 
    y = librosa.effects.preemphasis(y)
    y = librosa.effects.trim(y)[0]  
    y = librosa.util.normalize(y) 
    return y, sr


def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())


recognizer = sr.Recognizer()


def voice_to_text():
    q = queue.Queue()
    with sd.InputStream(callback=callback):
        print("Listening...")
        while True:
            audio_data = q.get()
           
            audio_tensor = torch.from_numpy(audio_data.T).float()
            text = model.transcribe(audio_tensor, 16000)
            print(f"Transcription: {text}")

if __name__ == "__main__":
 
    model = torch.hub.load('snakers4/silero-models', 'silero_stt', language='en')

    with sr.Microphone() as source:
        print("Speek something")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print(f"Google speech recognition: {text}")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not found")
        except sr.RequestError as e:
            print(f"Could not found request results; {e}")

    #sample audio file
    audio_file = 'audio_file.wav'
    processed_audio, sample_rate = preprocess_audio(audio_file)
    audio_tensor = torch.from_numpy(np.expand_dims(processed_audio, axis=0)).float()

    text = model.transcribe(audio_tensor, sample_rate)
    print(f"Transcription: {text}")

   
    voice_to_text()