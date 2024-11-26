import pyttsx3

engine = pyttsx3.init()


voices = engine.getProperty('voices') 
engine.setProperty('voice', voices[1].id)   # changing index, changes voices. 1 for female
engine.say("Hello world, This is Abishek K")
engine.runAndWait()