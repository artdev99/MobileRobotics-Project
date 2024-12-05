import thymiodirect
import asyncio


async def turn_off_all_leds():
    async with thymiodirect.ThymioAsyncClient() as client:
        await client.connect()  # Ensure your robot is correctly paired (e.g., via Bluetooth)

        # Set all LED variables to 0
        await client.set_variable("leds.top", [0, 0, 0])  # Turn off the top RGB LED
        await client.set_variable(
            "leds.bottom.left", [0, 0, 0]
        )  # Turn off the left bottom LED
        await client.set_variable(
            "leds.bottom.right", [0, 0, 0]
        )  # Turn off the right bottom LED
        await client.set_variable(
            "leds.circle", [0] * 8
        )  # Turn off all the circle LEDs
        await client.set_variable(
            "leds.prox.horiz", [0] * 5
        )  # Turn off proximity sensor LEDs
        await client.set_variable(
            "leds.prox.grids", [0, 0]
        )  # Turn off the ground sensor LEDs
        await client.set_variable("leds.rc", 0)  # Turn off remote control LED
        await client.set_variable("leds.temperature", 0)  # Turn off temperature LED
        await client.set_variable("leds.sound", 0)  # Turn off sound LED
        await client.set_variable("leds.buttons", [0] * 4)  # Turn off buttons LEDs
        await client.set_variable("leds.battery", 0)  # Turn off battery LED

        print("All LEDs are turned off.")


# Run the asyncio loop
asyncio.run(turn_off_all_leds())
