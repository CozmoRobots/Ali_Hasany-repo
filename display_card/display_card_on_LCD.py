import cozmo
import asyncio
from PIL import Image
import glob

# ############################### READ THIS !!! #############################################
# This script is just to test whether or not the cozmo displays the card suit and the card
# value correctly on its display it iterates through all the 52 playig cards
#############################################################################################

# collect all file names as a list
bmp_files = glob.glob('*.bmp')

# Function to display BMP image on Cozmo's face
def display_bmp_image(robot: cozmo.robot.Robot, bmp_image_path):
    # Load the BMP image using PIL
    img = Image.open(bmp_image_path)

    # Convert the image to a format suitable for Cozmo's face
    screen_data = cozmo.oled_face.convert_image_to_screen_data(img)

    # Display the image on Cozmo's face with a duration
    robot.display_oled_face_image(screen_data, duration_ms=2000)

# Main Cozmo program
async def main(robot: cozmo.robot.Robot):
    
    for i in ['hearts.bmp','spades.bmp','diamonds.bmp','clubs.bmp']:
        for j in ['A.bmp','2.bmp','3.bmp','4.bmp','5.bmp','6.bmp','7.bmp','8.bmp','9.bmp','10.bmp','J.bmp','Q.bmp','K.bmp']:
            bmp_suit_path = i
            bmp_value_path = j
            display_bmp_image(robot, bmp_suit_path)
            await asyncio.sleep(2)  # Wait for 2 seconds 
            display_bmp_image(robot, bmp_value_path)
            await asyncio.sleep(2)  # Wait for 2 seconds

cozmo.run_program(main)