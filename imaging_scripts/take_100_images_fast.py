import cozmo
from PIL import Image
import asyncio

# ############################### READ THIS !!! ##################################
#  This script will take photos siginificantly faster then take_100_images_slow.py
##################################################################################

async def take_and_save_images(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True

    # This will take 1 to 100 images and save them as 1.png to 100.png
    for i in range(1, 101):
        new_im = await robot.world.wait_for(cozmo.world.EvtNewCameraImage)
        img_latest = new_im.image.raw_image
        img_convert = img_latest.convert('L')
        img_convert.save(f'{i}.png')

        print(f"Image {i} captured and saved.")
        await asyncio.sleep(0.3)  # Adjust the delay between captures if needed

async def cozmo_program(robot: cozmo.robot.Robot):
    await take_and_save_images(robot)

cozmo.run_program(cozmo_program)