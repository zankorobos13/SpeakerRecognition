import asyncio
import json
import wave
import base64
import websockets
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
print(api_key)

WS_URL = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={api_key}"
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
VOICES_NAMES = ["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe"]
TEXTS_TO_SPEAK = None
SPEEDS_TO_SPEAK = ["[Быстрым голосом]", "[Медленным голосом]", ""]

async def get_wav_from_api(voice, speed, phrase, i, ws):
    await ws.send(json.dumps({
        "clientContent": {
            "turns": [{
                "role": "user",
                "parts": [{"text": f"Произнеси следующую фразу {speed} не добавляя ничего лишнего: {phrase}"}]
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
    
    with wave.open(f"data/{voice}/{i}.wav", "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000) 
            wf.writeframes(pcm_data)


async def main():
    with open("data/data.txt", "r", encoding="utf-8") as df:
        TEXTS_TO_SPEAK = df.read().split("\n")
    
    for i in range(2, len(VOICES_NAMES)):
        voice = VOICES_NAMES[i]
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
            for j in range(len(TEXTS_TO_SPEAK) // len(VOICES_NAMES)):
                speed = SPEEDS_TO_SPEAK[(j * len(VOICES_NAMES) + i) % len(SPEEDS_TO_SPEAK)]
                phrase = TEXTS_TO_SPEAK[j * len(VOICES_NAMES) + i]
                print(f"{j * len(VOICES_NAMES) + i} {speed} {voice}")
                await get_wav_from_api(voice=voice, speed=speed, phrase=phrase, i=j * len(VOICES_NAMES) + i, ws=ws)
                await asyncio.sleep(1) 

if __name__ == "__main__":
    asyncio.run(main())