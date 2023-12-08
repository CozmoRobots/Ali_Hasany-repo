import numpy as np
import pandas as pd
import random
import cozmo
from PIL import Image
import asyncio
import cozmo
import cv2
from PIL import Image, ImageDraw, ImageFont
from statistics import median
import tensorflow as tf
from collections import Counter
from tensorflow.keras.models import load_model
import time
import torch

# uncommmnet the following when using in actual gameplay with other players!
import socket
import errno
from socket import error as socket_error

# appending file path to include file that displays the card on the screen
import sys
sys.path.append('../display_card')

def display_bmp_image(robot: cozmo.robot.Robot, bmp_image_path):
    # Load the BMP image using PIL
    img = Image.open(bmp_image_path)

    # Convert the image to a format suitable for Cozmo's face
    screen_data = cozmo.oled_face.convert_image_to_screen_data(img)

    # Display the image on Cozmo's face with a duration of 2000 ms
    robot.display_oled_face_image(screen_data, duration_ms=2000) 

# Load the TensorFlow model which recognonizes the cards
model = load_model('best_model_2.h5')

# labels that will be used for the classification
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

# Load the saved pytorch model for predictions
# Dropping 'LABEL','result', 'dealer_value', 'dealer_bust' from our training data
# All the other variables are used as input features
# num_of_player,num_of_decks,big_card_count,small_card_count,dealer_card,init_hand,hit
# LABEL is the output feature it is basically the outcome variable which determines whether or not we should have hit/stayed
# Defining the Neural Network Model

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(7, 100)  # one-hot encoded features were removed
        #self.fc2 = nn.Linear(100, 200)
        self.fc3 = torch.nn.Linear(100, 20)  
        self.fc4 = torch.nn.Linear(20, 1)  # Output has 1 neuron for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


hit_stay_model = Net()
hit_stay_model.load_state_dict(torch.load('ali_model_cc.pt'))
hit_stay_model.eval()


def get_suit_value(card):
	if card == 'joker':
		return 'J', 'spades'
	else:
		my_card = card.split()
		if my_card[0] == 'ace':
			my_card[0] = 'A'
			return my_card[0], my_card[2]
		elif my_card[0] == 'two':
			my_card[0] = '2'
			return my_card[0], my_card[2]
		elif my_card[0] == 'three':
			my_card[0] = '3'
			return my_card[0], my_card[2]
		elif my_card[0] == 'four':
			my_card[0] = '4'
			return my_card[0], my_card[2]
		elif my_card[0] == 'five':
			my_card[0] = '5'
			return my_card[0], my_card[2]
		elif my_card[0] == 'six':
			my_card[0] = '6'
			return my_card[0], my_card[2]
		elif my_card[0] == 'seven':
			my_card[0] = '7'
			return my_card[0], my_card[2]
		elif my_card[0] == 'eight':
			my_card[0] = '8'
			return my_card[0], my_card[2]
		elif my_card[0] == 'nine':
			my_card[0] = '9'
			return my_card[0], my_card[2]
		elif my_card[0] == 'ten':
			my_card[0] = '10'
			return my_card[0], my_card[2]
		elif my_card[0] == 'jack':
			my_card[0] = 'J'
			return my_card[0], my_card[2]
		elif my_card[0] == 'queen':
			my_card[0] = 'Q'
			return my_card[0], my_card[2]
		elif my_card[0] == 'king':
			my_card[0] = 'K'
			return my_card[0], my_card[2]
		

def get_card(robot: cozmo.robot.Robot):
	# Set Cozmo's camera to stream in color
	robot.camera.color_image_enabled = True
	 #async def process_frame():
	robot.set_head_angle(cozmo.util.degrees(35)).wait_for_completed()
	_ = 0 # _ to count the number of iterations
	while _ < 10:
		# Get the latest image from Cozmo's camera
		iterations = 20
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
	return labels[most_common_class]



# make the shoe for blackjack
def make_shoe(num_decks, card_types):
	new_shoe = []
	for i in range(num_decks):
		for j in range(4):
			new_shoe.extend(card_types)
	random.shuffle(new_shoe)
	return new_shoe

# funciton to get the big and small card counts
def get_big_small_card_count(card_count):
	big_cards = card_count['K'] + card_count['A'] + card_count['J'] + card_count['Q'] + card_count['10']
	small_cards = card_count['2'] + card_count['3'] + card_count['4'] + card_count['5'] + card_count['6'] + card_count['7']
	
	return big_cards, small_cards
	
# update shoe after one card is removed
def remove_card_from_shoe(value_to_remove, input_list):
	# Check if the value to remove exists in the list
	if value_to_remove in input_list:
		# Find the index of the first occurrence of the value
		idx = input_list.index(value_to_remove)
		
		# Remove the value from the list at the found index
		input_list.pop(idx)
		
		return input_list
	
	else:
		# If the value doesn't exist in the list, return the original list
		return input_list

# to keep track of all the cards that leave the shoe
def increment_value(key, card_count_dict):
	# Check if the key exists in the dictionary
	if key in card_count_dict:
		card_count_dict[key] += 1  # Increment the value corresponding to the key by 1
	else:
		print(f"Key '{key}' not found in the dictionary.")

	return card_count_dict

def calculate_handval(val1, val2, i_hand):
	card_val = {'0':0, 'A':11, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':10, 'Q':10, 'K':10} 
	hand_value = card_val[val1]+card_val[val2]+i_hand
	if hand_value == 21:
		black_jack = 1
		cozmo_bust = 0
		return black_jack, cozmo_bust, hand_value
	elif hand_value > 21:
		black_jack = 0
		cozmo_bust = 1
		return black_jack, cozmo_bust, hand_value
	elif hand_value < 21:
		black_jack = 0
		cozmo_bust = 0
		return black_jack, cozmo_bust, hand_value


# we need the numerical value of the dealer's first card because our model accepts a numerical value not a string
def get_dealer_card_numerical_value(value):
	card_val = {'0':0, 'A':11, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':10, 'Q':10, 'K':10}
	return card_val[value]


# function to classify the raw score that is output from the hit/stay decision ML model
def binary_classification(output_value):
	threshold = 0.5
	if output_value >= threshold:
		return 1
	else:
		return 0
	

# I wanted to make a recursive function that would keep hitting until it reached the base case of staying but maybe will implement this
# some other time when I have more free time to play with this	
# def make_predictions(robot, num_of_player,num_of_decks,big_card_count,small_card_count,dealer_card,init_hand,hit):
# 	# Make predictions using the loaded model
# 	# write this down as a recursive function
# 	input_data = np.array([num_of_player,num_of_decks,big_card_count,small_card_count,dealer_card,init_hand,hit])
# 	# Reshape the input_data to match the expected input shape (assuming 7 features)
# 	input_data = np.reshape(input_data, (1, -1))  # Reshape to (1, 7)

# 	# Convert the input data to a PyTorch tensor
# 	input_tensor = torch.tensor(input_data, dtype=torch.float32)
# 	# Make predictions using the loaded model
# 	with torch.no_grad():
# 		predictions = hit_stay_model(input_tensor)
		
# 	hit_or_stay = str(binary_classification(predictions.item()))
# 	print(f'Hit or Stay decision: {hit_or_stay}')
# 	if hit_or_stay == '0':
# 		time.sleep(3)
# 		robot.say_text('Stay').wait_for_completed()
# 		time.sleep(3)
# 		hit = int(hit_or_stay)
# 		return hit_or_stay
# 	elif hit_or_stay == '1':
# 		time.sleep(3)
# 		robot.say_text('Hit').wait_for_completed()
# 		time.sleep(3)
# 		hit = int(hit_or_stay)
# 		make_predictions(robot, num_of_player,num_of_decks,big_card_count,small_card_count,dealer_card,init_hand,hit)

def make_predictions(num_of_player,num_of_decks,big_card_count,small_card_count,dealer_card,init_hand,hit):
	# Make predictions using the loaded model
	input_data = np.array([num_of_player,num_of_decks,big_card_count,small_card_count,dealer_card,init_hand,hit])
	# Reshape the input_data to match the expected input shape (assuming 7 features)
	input_data = np.reshape(input_data, (1, -1))  # Reshape to (1, 7) as I have 7 features

	# Convert the input data to a PyTorch tensor
	input_tensor = torch.tensor(input_data, dtype=torch.float32)
	# Make predictions using the loaded model
	with torch.no_grad():
		predictions = hit_stay_model(input_tensor)
		
	hit_or_stay = str(binary_classification(predictions.item()))
	return hit_or_stay

card_types = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']


def cozmo_program(robot: cozmo.robot.Robot):
	# starting off with some dummy values which will be changed overtime start the simulation
	# num_of_player,num_of_decks,big_card_count,small_card_count,dealer_card,init_hand,hit
	game = 1
	num_of_player = 4
	num_of_decks = 6
	big_card_count = 100
	small_card_count = 0
	dealer_card = 7
	init_hand = 0
	hit = 0

	card_count = {'A':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, 'J':0, 'Q':0, 'K':0} 

	shoe = make_shoe(num_of_decks, card_types)

	# uncomment the code below when playing the game with other players!
	# ############################### STEP 0 OF THE GAME: SET UP THE NETWORK TO RECEIVE DATA #################################
	# try:
	# 	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# except socket_error as msg:
	# 	print('************* socket FAILED ****************** ' + msg)
	# ip = "10.0.1.10"
	# port = 5000

	# try:
	# 	s.connect((ip, port))
	# except socket_error as msg:
	# 	print('*************** socket FAILED to bind ******************* ')
		
	# my_name = 'ALI_H' # my cozmo's name

	while game != 0 :
		
		# ############################### STEP 1 OF THE GAME: DEALER SHOWS FIRST CARD #################################

		# 1. dealer starts the game by showing first card
		#dealer_card_value_1 = get_dealer_card_1(robot)

		# instead of get_dealer_card_1 we can write the code here to get the data form the network like below:
		# bytedata = s.recv(4048)
		# data = bytedata.decode('utf-8')
		# print(data)
		# player_dealer_card_info = data.split(';')
		# dealer_card_value_1 = player_dealer_card_info[0]

		# 2. updating the card count by adding 1 to the value of the key which is the same as the dealer card 
		#card_count = increment_value(dealer_card_value_1, card_count)

		# 3. removing the dealer card from the shoe and updating the shoe
		#shoe = remove_card_from_shoe(dealer_card_value_1, shoe)

		# 4. keep track of big and small card

		# dealer_card is assinged a numerial value based on the value of dealer_card_value_1

		# ############################### STEP 2 OF THE GAME: COZMO GETS FIRST CARD #################################

		# This program will simulate the cozmo taking pictures and returning the card suit and value

		print("******************* CALLING FUNCTION TO TAKE FIRST IMAGE ********************")
		print("Place the card in the Cozmo's camera's range. You have 5 seconds! ")
		time.sleep(5)
		cozmo_card_1 = get_card(robot)
		print(f'******************* CARD 1 : {cozmo_card_1} ********************')
		print("******************* FUNCTION ENDED ********************")

		# 1. get the card suit and value of the first card dealt to my cozmo
		cozmo_card_value_1, suit_1 = get_suit_value(cozmo_card_1)
		robot.say_text(cozmo_card_1).wait_for_completed()

		# 2. updating the card count by adding 1 to the value of the key which is the same as the cozmo's first card 
		card_count = increment_value(cozmo_card_value_1, card_count)

		# 3. removing the card from the shoe and updating the shoe
		shoe = remove_card_from_shoe(cozmo_card_value_1, shoe)

		bmp_suit_path = f'..\\display_card\\{suit_1}.bmp'
		bmp_value_path = f'..\\display_card\\{cozmo_card_value_1}.bmp'

		# display the card on the screen
		display_bmp_image(robot, bmp_suit_path)
		time.sleep(2)  # Wait for 2 seconds (adjust as needed)
		display_bmp_image(robot, bmp_value_path)
		time.sleep(2)  # Wait for 2 seconds (adjust as needed)
		

		# ############################### STILL STEP 2 OF THE GAME !!! COZMO GETS SECOND CARD #################################

		print("******************* CALLING FUNCTION TO TAKE SECOND IMAGE ********************")
		print("Place the card in the Cozmo's camera's range. You have 5 seconds! ")
		time.sleep(5)
		cozmo_card_2 = get_card(robot)
		print(f'******************* CARD 2: {cozmo_card_2} ********************')
		print("******************* FUNCTION ENDED ********************")

		# 1. get the card suit and value of the second card dealt to my cozmo
		cozmo_card_value_2, suit_2 = get_suit_value(cozmo_card_2)
		robot.say_text(cozmo_card_2).wait_for_completed()
		# 2. updating the card count by adding 1 to the value of the key which is the same as the cozmo's second card 
		card_count = increment_value(cozmo_card_value_2, card_count)

		# 3. removing the card from the shoe and updating the shoe
		shoe = remove_card_from_shoe(cozmo_card_value_2, shoe)

		bmp_suit_path = f'../display_card/{suit_2}.bmp'
		bmp_value_path = f'../display_card/{cozmo_card_value_2}.bmp'

		# display the card on the screen
		display_bmp_image(robot, bmp_suit_path)
		time.sleep(2)  # Wait for 2 seconds (adjust as needed)
		display_bmp_image(robot, bmp_value_path)
		time.sleep(2)  # Wait for 2 seconds (adjust as needed)

		# 4. calculating the intial hand value based off the two cards
		cozmo_black_jack, cozmo_bust, init_hand = calculate_handval(cozmo_card_value_1, cozmo_card_value_2, init_hand) 

		# 5. Keeping the card count based on my technique
		big_card_count, small_card_count = get_big_small_card_count(card_count)

		time.sleep(7)  # Wait for 7 seconds (adjust as needed)
		robot.say_text(f'hand value is {init_hand}').wait_for_completed()
		time.sleep(5)  # Wait for 5 seconds (adjust as needed)
		# ############################### STEP 3 OF THE GAME: GET OTHER PLAYERS' CARD INFO #################################
		# 1. get all players' card info
		# 2. pop out cards from the shoe
		# 3. increment card count value
		# 4. keep track of big and small card count


		# Implement a for loop and pass it the number of players in the game

		# get the data as follows:
		# bytedata = s.recv(4048)
		# data = bytedata.decode('utf-8')
		# print(data)
		# player_dealer_card_info = data.split(';')
		# player_1_card = player_dealer_card_info[1]
		# player_2_card = player_dealer_card_inf[2]


		# ############################### STEP 4 OF THE GAME MAKE HIT and STAY DECISION #################################
		if init_hand < 17 and cozmo_bust == 0 and cozmo_black_jack != 1:
			hit = 1
			robot.say_text("hit").wait_for_completed()
			time.sleep(5)  # Wait for 2 seconds (adjust as needed)


			print("******************* CALLING FUNCTION TO TAKE THIRD IMAGE ********************")
			print("Place the card in the Cozmo's camera's range. You have 5 seconds! ")
			time.sleep(5)
			cozmo_card_3 = get_card(robot)
			print(f'******************* CARD 3: {cozmo_card_3} ********************')
			print("******************* FUNCTION ENDED ********************")

			# 1. get the card suit and value of the third card dealt to my cozmo
			cozmo_card_value_3, suit_3 = get_suit_value(cozmo_card_3)
			robot.say_text(cozmo_card_3).wait_for_completed()
			# 2. updating the card count by adding 1 to the value of the key which is the same as the cozmo's third card 
			card_count = increment_value(cozmo_card_value_3, card_count)

			# 3. removing the card from the shoe and updating the shoe
			shoe = remove_card_from_shoe(cozmo_card_value_3, shoe)

			bmp_suit_path = f'../display_card/{suit_3}.bmp'
			bmp_value_path = f'../display_card/{cozmo_card_value_3}.bmp'

			# display the card on the screen
			display_bmp_image(robot, bmp_suit_path)
			time.sleep(2)  # Wait for 2 seconds (adjust as needed)
			display_bmp_image(robot, bmp_value_path)
			time.sleep(2)  # Wait for 2 seconds (adjust as needed)

			# 4. calculating the intial hand value based off the two cards
			cozmo_black_jack, cozmo_bust, init_hand = calculate_handval(cozmo_card_value_3, '0', init_hand) 

			# 5. Keeping the card count based on my technique
			big_card_count, small_card_count = get_big_small_card_count(card_count)


			time.sleep(7)  # Wait for 7 seconds (adjust as needed)
			robot.say_text(f'hand value is {init_hand}').wait_for_completed()
			time.sleep(5)  # Wait for 5 seconds (adjust as needed)			
		else:
			hit = 0
			time.sleep(7)  # Wait for 7 seconds (adjust as needed)
			robot.say_text('Stay').wait_for_completed()
			time.sleep(5)
		
		# there is no reason to hit again if  previous deciosion is stay
		prev_hit = hit
		# Organize the input variables into an array or list
		# or write this as a recursive function (it was becoming too complex! I can try this some later time!)
		hit_or_stay = make_predictions(num_of_player,num_of_decks,big_card_count,small_card_count,dealer_card,init_hand,hit)
		hit = int(hit_or_stay) # for next time when the NN is used

		if prev_hit == 1 and hit_or_stay == '1' and cozmo_bust == 0 and cozmo_black_jack != 1:
			time.sleep(2)
			robot.say_text("hit").wait_for_completed()
			time.sleep(5)  # Wait for 2 seconds (adjust as needed)


			print("******************* CALLING FUNCTION TO TAKE FOURTH IMAGE ********************")
			print("Place the card in the Cozmo's camera's range. You have 5 seconds! ")
			time.sleep(5)
			cozmo_card_4 = get_card(robot)
			print(f'******************* CARD 4: {cozmo_card_4} ********************')
			print("******************* FUNCTION ENDED ********************")

			# 1. get the card suit and value of the 4th card dealt to my cozmo
			cozmo_card_value_4, suit_4 = get_suit_value(cozmo_card_4)
			robot.say_text(cozmo_card_4).wait_for_completed()
			# 2. updating the card count by adding 1 to the value of the key which is the same as the cozmo's 4th card 
			card_count = increment_value(cozmo_card_value_4, card_count)

			# 3. removing the card from the shoe and updating the shoe
			shoe = remove_card_from_shoe(cozmo_card_value_4, shoe)

			bmp_suit_path = f'../display_card/{suit_4}.bmp'
			bmp_value_path = f'../display_card/{cozmo_card_value_4}.bmp'

			# display the card on the screen
			display_bmp_image(robot, bmp_suit_path)
			time.sleep(2)  # Wait for 2 seconds (adjust as needed)
			display_bmp_image(robot, bmp_value_path)
			time.sleep(2)  # Wait for 2 seconds (adjust as needed)

			# 4. calculating the intial hand value based off the four cards
			cozmo_black_jack, cozmo_bust, init_hand = calculate_handval(cozmo_card_value_4, '0', init_hand) 

			# 5. Keeping the card count based on my technique
			big_card_count, small_card_count = get_big_small_card_count(card_count)

			time.sleep(7)  # Wait for 7 seconds (adjust as needed)
			robot.say_text(f'hand value is {init_hand}').wait_for_completed()
			time.sleep(5)  # Wait for 5 seconds (adjust as needed)

		else:
			hit = 0
			time.sleep(5)
			robot.say_text('Stay').wait_for_completed()
			time.sleep(5)


		# ############################### STEP 5 OF THE GAME: GET DEALER 2nd card info #################################
		# 1. get dealer's second card info
		# 2. pop out card from the shoe
		# 3. increment card count value
		# 4. keep track of big and small card counts 
		# 5. calculate dealer hand value and check dealer_bust




		# ############################### STEP 6 OF THE GAME: ROUND WON OR LOST OR DRAW #################################
		# 1. compare dealer bust, cozmo bust AND/OR
		# 2. compare dealer blackjack or cozmo blackjack
		# 3. compare dealer hand and cozmo hand



		# restarting for next round
		init_hand = 0
		cozmo_black_jack = 0
		cozmo_bust = 0
		dealer_bust = 0
		dealer_card = 0


		# ending game if less than ten cards in the shoe
		if len(shoe) < 10:
			game = 0



cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=False)