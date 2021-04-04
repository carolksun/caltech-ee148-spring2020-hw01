from find_bounding_box import detect_red_light
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
mpl.use('tkagg')

width = 29
height = 66

def display_image_with_bounding_box(file_name, boxes):
    im = Image.open(file_name)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(im)

    for box in boxes:
        ax.add_patch(patches.Rectangle((box[0], box[1]), width, height, linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()

# Q5
for index in ['010', '011', '071', '160']:
    file_name = f'../data/RedLights2011_Medium/RL-{index}.jpg'
    I = Image.open(file_name)
    I = np.asarray(I)
    box = detect_red_light(I)
    display_image_with_bounding_box(file_name, box)

# Q6
# for index in ['247', '012', '064', '182']:
#     file_name = f'../data/RedLights2011_Medium/RL-{index}.jpg'

#     tl = get_bb(file_name)
#     display_image_with_bounding_box(file_name, tl)