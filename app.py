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
from gtts import gTTS
from pathlib import Path
import click
import warnings
import time
import openai
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')
warnings.filterwarnings("ignore")

@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny", "base", "small", 
              "medium", "large"]))
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
    threading.Thread(target=reply, args=(result_queue,), daemon=True).start()

    try:
        while True:
            print(result_queue.get())
    except KeyboardInterrupt:
        print("\nExiting...")

def record_audio(audio_queue, energy, pause, dynamic_energy):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = energy
    recognizer.pause_threshold = pause
    recognizer.dynamic_energy_threshold = dynamic_energy

    try:
        with sr.Microphone(sample_rate=16000) as source:
            print("Listening...")
            while True:
                audio = recognizer.listen(source)
                torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                audio_queue.put_nowait(torch_audio)
    except Exception as e:
        print(f"Error in record_audio: {e}")

def transcribe_forever(audio_queue, result_queue, audio_model, english, wake_word, verbose):
    while True:
        try:
            audio_data = audio_queue.get()
            result = audio_model.transcribe(audio_data, language='english' if english else None)
            predicted_text = result["text"].strip()

            if re.search(rf"\b{re.escape(wake_word)}\b", predicted_text, re.IGNORECASE):
                cleaned_text = re.sub(rf"\b{re.escape(wake_word)}\b", "", predicted_text, flags=re.IGNORECASE).strip()
                cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)  # Remove punctuation
                if verbose:
                    print(f"Wake word detected. Processing: {cleaned_text}")
                result_queue.put_nowait(cleaned_text)
            elif verbose:
                print(f"Ignored: {predicted_text}")
        except Exception as e:
            print(f"Error in transcribe_forever: {e}")

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

# Reply to the user
def reply(result_queue, verbose):
    while True:
        question = result_queue.get()
        # We use the following format for the prompt: "Q: ?\nA:"
        prompt = "Q: {}?\nA:".format(question)
        
        data = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            temperature=0.5,
            max_tokens=100,
            n=1,
            stop=["\n"]
        )

        # We catch the exception in case there is no answer
        try:
            answer = data["choices"][0]["text"]
            mp3_obj = gTTS(text=answer, lang="en", slow=False)
        except Exception as e:
            choices = [
                "I'm sorry, I don't know the answer to that",
                "I'm not sure I understand",
                "I'm not sure I can answer that",
                "Please repeat the question in a different way"
            ]
            mp3_obj = gTTS(text=choices[np.random.randint(0, len(choices))], lang="en", slow=False)
            if verbose:
                print(e)

        # In both cases, we play the audio
        mp3_obj.save("reply.mp3")
        reply_audio = AudioSegment.from_mp3("reply.mp3")
        play(reply_audio)

if __name__ == "__main__":
    main()
