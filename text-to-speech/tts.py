from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import optparse
import os

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS"), )

audio = client.text_to_speech.convert(
    text="pallavi is my babeji",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

play(audio)
