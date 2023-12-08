import os

# #################################### READ THIS !!! #############################################
# This script creates empty directories needed to store the images for later use in training the 
# ML model for recognizing the card suit and value
##################################################################################################


# Function to create a directories
def create_directory(directory_name):
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_name}' already exists.")
    except Exception as e:
        print(f"Failed to create directory '{directory_name}': {e}")

for i in ['hearts','spades','clubs','diamonds']:
    for j in ['ace','two','three','four','five','six','seven','eight','nine','ten','jack','queen','king']:
        create_directory(f'{j} of {i}')