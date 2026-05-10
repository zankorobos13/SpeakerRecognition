import asyncio
import json
import wave
import base64
import websockets
from dotenv import load_dotenv
import os
import random

load_dotenv()
api_key = os.getenv("API_KEY")
print(api_key)

WS_URL = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={api_key}"
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
VOICES_NAMES = ["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe"]
TEXTS_TO_SPEAK = None
SPEEDS_TO_SPEAK = ["[Быстрым голосом]", "[Медленным голосом]", ""]

async def get_wav_from_api(voice, phrase, ws):
    await ws.send(json.dumps({
        "clientContent": {
            "turns": [{
                "role": "user",
                "parts": [{"text": f"Произнеси следующую фразу не добавляя ничего лишнего: {phrase}"}]
            }],
            "turnComplete": True
        }
    }))

    pcm_data = b""

    async for message in ws:
        data = json.loads(message)

        parts = (data.get("serverContent", {}).get("modelTurn", {}).get("parts", []))

        for part in parts:
            if "inlineData" in part:
                b64_audio = part["inlineData"]["data"]
                audio_bytes = base64.b64decode(b64_audio)
                pcm_data += audio_bytes

        # можно остановиться после завершения ответа
        if data.get("serverContent", {}).get("turnComplete"):
            break
    
    with wave.open(f"data/_tests/{random.randint(0, 99999)}.wav", "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000) 
            wf.writeframes(pcm_data)


async def main(phrase, voice):    
    async with websockets.connect(WS_URL) as ws:
        print(f"Connected {voice}")

        await ws.send(json.dumps({
            "setup": {
                "model": MODEL,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": voice
                            }
                        }
                    }
                }
            }
        }))
        await get_wav_from_api(voice=voice, phrase=phrase, ws=ws)
            

if __name__ == "__main__":
    asyncio.run(main("А сейчас проверяем как говорит Callirrhoe будет ли правильное предсказание", "Callirrhoe"))