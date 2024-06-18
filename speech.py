import whisper
import pyaudio
import numpy as np
import wave


model = whisper.load_model("base")


FORMAT = pyaudio.paInt16  
CHANNELS = 1              
RATE = 16000              
CHUNK = 1024              
RECORD_SECONDS = 10        


audio = pyaudio.PyAudio()


stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Listening...")


def transcribe_audio(audio_data):
    audio_array = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0
    audio_segment = whisper.pad_or_trim(audio_array)
    mel = whisper.log_mel_spectrogram(audio_segment).to(model.device)
    options = whisper.DecodingOptions(language="en", fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text


def save_audio_to_file(audio_data, filename="debug_audio.wav"):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(audio_data)
    wf.close()

try:
    while True:
        print("Recording for {} seconds...".format(RECORD_SECONDS))
        frames = []
        
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        
        audio_data = b''.join(frames)
        
       
        save_audio_to_file(audio_data, "debug_audio.wav")
        
        
        text = transcribe_audio(audio_data)
        print(f"Transcribed Text: {text}")

except KeyboardInterrupt:
    print("Stopping...")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
