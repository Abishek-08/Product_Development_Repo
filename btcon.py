import asyncio
from bleak import BleakScanner

async def discover_devices():
    devices = await BleakScanner.discover()
    for device in devices:
        print(f"Device: {device.name}, Address: {device.address}")
        return device.address
# Run the discover function
asyncio.run(discover_devices())
