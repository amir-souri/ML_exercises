# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Get the input filename
filename = "./inputs/zico.jpg"

# Loads a gray-scale image from a file passed as argument.
image = cv2.imread(filename, cv2.IMREAD_COLOR)

# Create the Matplotlib figure.
fig = plt.figure("Images")

# This function creates a Matplotlib window and shows four images.
def show_image(image, pos, title="Image", isGray=False):
    sub = fig.add_subplot(1, 3, pos)
    #add_subplot(nrows, ncols,
    #*pos* is a three digit integer, where the first digit is the
            # number of rows, the second the number of columns, and the third
            # the index of the subplot. i.e. fig.add_subplot(235) is the same as
            # fig.add_subplot(2, 3, 5). Note that all integers must be less than
            # 10 for this form to work.
    sub.set_title(title)
    sub.imshow(image)
    plt.axis("off")
    if isGray:
        sub.imshow(image, cmap="gray")
    else:
        sub.imshow(image)

# <Exercise 2.6>

# <Exercise 2.6 (a)>
mask = np.zeros(image.shape, dtype=np.uint8)
# Construct a mask to display the interested regions.
# <Exercise 2.6 (b)>
# rows,cols,channels = image.shape
rec1 = cv2.rectangle(img=mask, pt1=(160,60), pt2=(320,180), color=(255,255,255), thickness=-1)  #(255,255,255) is withe
#Set the thickness as − 1 (or any negative number) to draw a filled white rectangle in the mask.
rec2 = cv2.rectangle(img=mask, pt1=(640,60), pt2=(800,180), color=(255,255,255), thickness=-1)  #(255,255,255) is withe

# <Exercise 2.6 (c)>
cir = cv2.circle(img=mask, center=(540,620), radius=60, color=(255,255,255), thickness=-1 )

show_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1, title="input")
show_image(cv2.cvtColor(cir, cv2.COLOR_BGR2RGB), 2, title="Mask")
# <Exercise 2.6 (d)>
result = cv2.bitwise_and(image, mask)
show_image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), 3, title="Result")

# Show the Matplotlib windows.
plt.show()
