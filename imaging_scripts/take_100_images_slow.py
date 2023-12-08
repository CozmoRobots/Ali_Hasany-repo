#import the cozmo and image libraries
import cozmo
from cozmo.objects import LightCube1Id, LightCube2Id, LightCube3Id
from PIL import Image
#import libraries for movement and asynchronous behavior
import asyncio
from cozmo.util import degrees, distance_mm

#import these libraries when needed for threads
import _thread
import time


# ############################### READ THIS !!! ################################
# This script is slow for taking pictures but I used this to take images in 
# which I had to significantly alter cozmos postion between each take
# this script gave me enough time to move cozmo
################################################################################


def cozmo_program(robot: cozmo.robot.Robot):
	
	success = True
	
	
	robot.camera.image_stream_enabled = True
	

	
  
	# this will take 100 pictures and name them 1.png to 100.png
	for i in range(1,101):
		new_im = robot.world.wait_for(cozmo.world.EvtNewCameraImage)
		new_im.image.raw_image.show()
	
		#save the raw image as a bmp file
		img_latest = robot.world.latest_image.raw_image
		img_convert = img_latest.convert('L')
		img_convert.save(f'{i}.png')
	
		#save the raw image data as a png file, named imageName
		# imageName = f'{i}.png'
		# img = Image.open("aPhoto.bmp")
		# width, height = img.size
		# new_img = img.resize( (width, height) )
		# new_img.save( imageName, 'png')
	
	
	return

cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)
