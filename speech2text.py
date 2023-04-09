import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()
# Set the voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Use the second voice in the list
# Set the volume
engine.setProperty('volume', 1.0)  # Set the volume to maximum

def speak(dialogue, rate=200):
    engine.setProperty('rate', rate) # Set the speaking rate
    engine.say(dialogue)
    # Run the TTS engine
    engine.runAndWait()
