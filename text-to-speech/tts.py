from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play, save
import optparse
import os

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS"), )

audio = client.text_to_speech.convert(
    text="This is a test audio",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

# play(audio)
save(audio, './text-to-speech/audio.wav')
