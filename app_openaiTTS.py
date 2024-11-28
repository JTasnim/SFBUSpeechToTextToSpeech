from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
import whisper
import queue
import os
import threading
import torch
import numpy as np
import re
from pathlib import Path
import click
import warnings
import openai
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv() 
openai.api_key = os.getenv("OPENAI_API_KEY")
warnings.filterwarnings("ignore")

# CLI Options
@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny", "base", "small", "medium", "large"]))
@click.option("--english", default=False, help="Whether to use the English model", is_flag=True, type=bool)
@click.option("--energy", default=300, help="Energy level for the mic to detect", type=int)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--dynamic_energy", default=False, is_flag=True, help="Flag to enable dynamic energy", type=bool)
@click.option("--wake_word", default="hey computer", help="Wake word to listen for", type=str)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True, type=bool)
def main(model, english, energy, pause, dynamic_energy, wake_word, verbose):
    if model != "large" and english:
        model = model + ".en"
    audio_model = whisper.load_model(model)
    audio_queue = queue.Queue()
    result_queue = queue.Queue()

    # Start threads
    threading.Thread(target=record_audio, args=(audio_queue, energy, pause, dynamic_energy), daemon=True).start()
    threading.Thread(target=transcribe_forever, args=(audio_queue, result_queue, audio_model, english, wake_word, verbose), daemon=True).start()
    threading.Thread(target=reply, args=(result_queue, verbose), daemon=True).start()

    while True:
        try:
            print(result_queue.get())
        except KeyboardInterrupt:
            print("\nExiting...")
            break

# Record Audio
def record_audio(audio_queue, energy, pause, dynamic_energy):
    r = sr.Recognizer()
    r.energy_threshold = energy
    r.pause_threshold = pause
    r.dynamic_energy_threshold = dynamic_energy

    try:
        with sr.Microphone(sample_rate=16000) as source:
            print("Listening...")
            while True:
                audio = r.listen(source)
                torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                audio_queue.put_nowait(torch_audio)
    except Exception as e:
        print(f"Error in record_audio: {e}")

# Transcribe Audio
def transcribe_forever(audio_queue, result_queue, audio_model, english, wake_word, verbose):
    wake_words = [wake_word, "computer", "hey, computer"]
    while True:
        try:
            audio_data = audio_queue.get()
            result = audio_model.transcribe(audio_data, language='english' if english else None)
            predicted_text = result["text"].strip()

            if any(w.lower() in predicted_text.lower() for w in wake_words):
                cleaned_text = re.sub(re.escape(wake_word), "", predicted_text, flags=re.IGNORECASE).strip()
                cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove punctuation
                if verbose:
                    print(f"Wake word detected. Processing: {cleaned_text}")
                result_queue.put_nowait(cleaned_text)
            else:
                if verbose:
                    print(f"Ignored: {predicted_text}")
        except Exception as e:
            print(f"Error in transcribe_forever: {e}")

# Get AI Completion
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=150):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error in get_completion_from_messages: {e}")
        return "I'm sorry, I couldn't process that."

# Reply with TTS
def reply(result_queue, verbose):
    while True:
        try:
            result = result_queue.get()
            messages = [{"role": "user", "content": result}]
            answer = get_completion_from_messages(messages)

            speech_file_path = Path(__file__).parent / "speech.mp3"
            try:
                response = openai.Audio.create(
                    model="tts-1",
                    voice="alloy",
                    input=answer
                )
                with open(speech_file_path, "wb") as f:
                    f.write(response['data'])

                reply_audio = AudioSegment.from_mp3(speech_file_path)
                print("Playing audio...")
                play(reply_audio)
                os.remove(speech_file_path)
            except Exception as e:
                print(f"Error in generating or playing audio: {e}")
        except Exception as e:
            print(f"Error in reply: {e}")

if __name__ == "__main__":
    main()
