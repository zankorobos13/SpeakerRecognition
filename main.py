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
VOICES_NAMES = ["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe"]
TEXT_TO_SPEAK = "Проверка произнесения текста более длинной фразы чем проверялось ранее чтобы проверить как работет чанкинг и тому подобное"

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
                                "voiceName": VOICES_NAMES[0]
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

        i = 0
        async for message in ws:
            data = json.loads(message)

            parts = (data.get("serverContent", {}).get("modelTurn", {}).get("parts", []))

            chunk = None

            for part in parts:
                if "inlineData" in part:
                    b64_audio = part["inlineData"]["data"]
                    audio_bytes = base64.b64decode(b64_audio)
                    chunk = audio_bytes

                    print("Получен кусок аудио")
            
            with open(f"audio/output{i}.pcm", "wb") as f:
                if chunk is not None:
                    f.write(chunk)
                    print(len(chunk))

            print(f"Аудио сохранено в output{i}.pcm")

            # можно остановиться после завершения ответа
            if data.get("serverContent", {}).get("turnComplete"):
                break

            i += 1
        
        

asyncio.run(main())