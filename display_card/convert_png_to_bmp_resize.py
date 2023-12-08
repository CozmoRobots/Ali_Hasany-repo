import glob
import os
from PIL import Image

# ############################### READ THIS !!! #############################################
# This script is needed to convert the png images to bmp images which is what cozmo's display
# in addition to resizing them to cozmo's display
#############################################################################################

# Import all PNG file names in a list
png_files = glob.glob('*.png')

# Function to convert a PNG image to a monochrome BMP image
def convert_to_monochrome_bmp(input_image_path, output_bmp_path):
    img = Image.open(input_image_path)
    img = img.convert('1')  # Convert to monochrome 1-bit image
    img = img.resize((128, 32), Image.BILINEAR)  # Resize to Cozmo's display size with bilinear interpolation because the other interpolations were not working
    img.save(output_bmp_path, format='BMP')  # Save as BMP format

# looping through all the png files
for i in png_files:
    print(f'Converting {i} to BMP format...')
    filename, extension = os.path.splitext(i)
    name_of_bmpfile = f"{filename}.bmp"
    convert_to_monochrome_bmp(i, name_of_bmpfile)
    print(f'Converted to {name_of_bmpfile}')