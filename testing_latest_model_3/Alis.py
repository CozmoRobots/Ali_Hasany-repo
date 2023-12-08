import cozmo
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import asyncio
from statistics import median
import tensorflow as tf
from collections import Counter
from tensorflow.keras.models import load_model

# Load the TensorFlow model
model = load_model('best_model_3.h5')

labels= ['ace of clubs',
'ace of diamonds',
 'ace of hearts',
 'ace of spades',
 'eight of clubs',
 'eight of diamonds',
 'eight of hearts',
 'eight of spades',
 'five of clubs',
 'five of diamonds',
 'five of hearts',
 'five of spades',
 'four of clubs',
 'four of diamonds',
 'four of hearts',
 'four of spades',
 'jack of clubs',
 'jack of diamonds',
 'jack of hearts',
 'jack of spades',
 'joker',
 'king of clubs',
 'king of diamonds',
 'king of hearts',
 'king of spades',
 'nine of clubs',
 'nine of diamonds',
 'nine of hearts',
 'nine of spades',
 'queen of clubs',
 'queen of diamonds',
 'queen of hearts',
 'queen of spades',
 'seven of clubs',
 'seven of diamonds',
 'seven of hearts',
 'seven of spades',
 'six of clubs',
 'six of diamonds',
 'six of hearts',
 'six of spades',
 'ten of clubs',
 'ten of diamonds',
 'ten of hearts',
 'ten of spades',
 'three of clubs',
 'three of diamonds',
 'three of hearts',
 'three of spades',
 'two of clubs',
 'two of diamonds',
 'two of hearts',
 'two of spades']

def cozmo_program(robot: cozmo.robot.Robot):
    # Set Cozmo's camera to stream in color
	robot.camera.color_image_enabled = True
	
	robot.set_head_angle(cozmo.util.degrees(+50)).wait_for_completed()
	_ = 0 # _ to count the number of iterations
	while True:
		
		iterations = 10
		preds = []
		for _ in range(iterations):
			robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=5)
			pil_image = robot.world.latest_image.raw_image.convert("RGB")

			# Convert PIL Image to numpy array
			image = np.array(pil_image)

			# Convert BGR image to RGB format
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



			bw = image[8:image.shape[0]-8, 48:image.shape[1]-48]

			prediction = model.predict(np.expand_dims(bw, axis=0))

			# Get the predicted class label
			predicted_class = np.argmax(prediction)
			preds.append(predicted_class)

		frequency = Counter(preds)
		most_common_class, _ =  frequency.most_common(1)[0]
		# Display the predicted class label on the image
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(bw, f'Prediction: {labels[most_common_class]}', (10, 30), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


		# print(bw.shape)
		cv2.imshow("Cozmo's View", bw)
		# Break loop on 'q' key
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cv2.destroyAllWindows()
        # Start processing frames
        #robot.loop.create_task(process_frame())

cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=False)
