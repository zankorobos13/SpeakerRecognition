import asyncio
import json
import base64
import websockets
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
print(api_key)

WS_URL = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={api_key}"
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
VOICE_NAME = "Zephyr"
TEXT_TO_SPEAK = "Проверка произнесения текста данного на вход"

async def main():
    async with websockets.connect(WS_URL) as ws:
        print("Connected")

        await ws.send(json.dumps({
            "setup": {
                "model": MODEL,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": VOICE_NAME
                            }
                        }
                    }
                }
            }
        }))

        await ws.send(json.dumps({
            "clientContent": {
                "turns": [{
                    "role": "user",
                    "parts": [{"text": f"Произнеси следующую фразу не добавляя ничего лишнего: {TEXT_TO_SPEAK}"}]
                }],
                "turnComplete": True
            }
        }))

        audio_chunks = []

        async for message in ws:
            data = json.loads(message)

            parts = (data.get("serverContent", {}).get("modelTurn", {}).get("parts", []))
            # print(parts)
            for part in parts:
                if "inlineData" in part:
                    b64_audio = part["inlineData"]["data"]
                    audio_bytes = base64.b64decode(b64_audio)
                    audio_chunks.append(audio_bytes)

                    print("Получен кусок аудио")

            # можно остановиться после завершения ответа
            if data.get("serverContent", {}).get("turnComplete"):
                break

        with open("output.pcm", "wb") as f:
            for chunk in audio_chunks:
                f.write(chunk)

        print("Аудио сохранено в output.pcm")

asyncio.run(main())