import asyncio
from bleak import BleakClient, BleakScanner
import pygame


# async def connect_and_play_audio():
#     # Initialize the mixer
#     pygame.mixer.init()

#     # Replace with your Bluetooth device's address
#     address = "8C:64:A2:8A:A7:4E"
#     AUDIO_FILE = "D:\Product Development\Object_Bottle_Buds\Calling-Santa.mp3"  # Path to your audio file

#     # Load and play the audio file
#     pygame.mixer.music.load(AUDIO_FILE)
#     pygame.mixer.music.play()

    
#     async with BleakClient(address) as client:
#         print(f"Connected to {client.address}")
        
#         # Play the audio file
#         # Keep the script running while the audio plays
#         while pygame.mixer.music.get_busy():
#             pygame.time.Clock().tick(10)
#         print("Audio playback finished.")

#         # Disconnect
#         print("Disconnected.")

# # Run the connect and play function
# asyncio.run(connect_and_play_audio())

device = "8C:64:A2:8A:A7:4E"


async def connect():
    async with BleakClient(device) as client:
        print(f"Connected to {client.address}")


asyncio.run(connect())

