import numpy as np
import pandas as pd
import random
#import cozmo
#from PIL import Image
#import asyncio
#import cozmo
#import cv2
#from PIL import Image, ImageDraw, ImageFont
#import asyncio
#from statistics import median
#import tensorflow as tf
#from collections import Counter
#from tensorflow.keras.models import load_model
import time



# Given list
my_list = ['A', 'B', 'C', 'A', 'D', 'D', 'D', 'A']

# Check if the list has at least two elements
if len(my_list) >= 2:
    # Choose two random indices
    indices = random.sample(range(len(my_list)), 2)
    print(type(indices))
    print(indices)
    # Get the values at the chosen indices
    values = [my_list[idx] for idx in indices]
    print(type(values))
    print(values)
    # Remove the chosen values from the list
    my_list = [value for value in my_list if value not in values]

    print("Updated list after removing chosen values:", my_list)
else:
    print("List doesn't have enough elements to pick two values.")


def calculate_handval(val1, val2):
	card_val = {'A':11, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, 'J':10, 'Q':10, 'K':10} 
	return card_val[val1]+card_val[val2]

val1 = 'A'
val2 = 'K'
print(calculate_handval(val1,val2))

val1 = '2'
val2 = '5'
print(calculate_handval(val1,val2))