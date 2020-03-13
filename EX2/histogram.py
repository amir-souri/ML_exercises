# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Get the input filename
filename = "./inputs/lena.jpg"

# Loads a gray-scale image from a file passed as argument.
image = cv2.imread(filename, cv2.IMREAD_COLOR)
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Create the Matplotlib figures.
fig_imgs = plt.figure("Images")
fig_hist = plt.figure("Histograms")


# This function creates a Matplotlib window and shows four images.
def show_image(image, pos, title="Image", isGray=False):
    sub = fig_imgs.add_subplot(2, 2, pos)
    sub.set_title(title)
    if isGray:
        sub.imshow(image, cmap="gray")
    else:
        sub.imshow(image)
    sub.axis("off")


# This function creates a Matplotlib window and shows four histograms.
def show_histogram(histogram, pos, title="Histogram"):
    sub = fig_hist.add_subplot(2, 2, pos)
    sub.set_title(title)
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    plt.xlim([0, 256])
    plt.ylim([0, 10000])
    plt.plot(histogram)


# <Exercise 2.5 (a)>
# Gray-scale image.

GrayscaleHistogram = cv2.calcHist(images=[grayscale], channels=[0], mask=None, histSize=[256], ranges=[0, 255])
show_histogram(GrayscaleHistogram, 1, title="Original Histogram")

# <Exercise 2.5 (b)>
# Shuffled image.
# TODO Why np.random.shuffle does not change the input at all! If we use random... the distance will be 0!
# The histogram should be the same. Since we are shuffling the pixel place not their values.
# np.random.shuffle : This function only shuffles the array along the first axis of a
# multi-dimensional array. The order of sub-arrays is changed but
# their contents remains the same.
np.random.shuffle(grayscale)
shuffleHistogram = cv2.calcHist(images=[grayscale], channels=[0], mask=None, histSize=[256], ranges=[0, 255])

show_histogram(shuffleHistogram, 2, title="Shuffled Histogram")


# <Exercise 2.5 (c)>
# Histogram distance
def calculate_histogram_distance(hist1, hist2):
    res = 0
    shape = hist1.shape
    print(shape[0])
    for i in range(shape[0]):
        res += (hist1[i][0] - hist2[i][0]) ** 2
    return res


def calculate_histogram_distance2(h1, h2):
    dis = 0
    print(len(h1))
    for i in range(len(h1)):
        dis += (h1[i] - h2[i]) ** 2

    return dis


# <Exercise 2.5 (d)>
# Calculate the distance between regular and shuffled image
# TODO Which function is more precise?  96681008.0     96680904.
# both there is probably just a rounding error
distance = calculate_histogram_distance(GrayscaleHistogram, shuffleHistogram)
print(f"4 {distance}")

distance2 = calculate_histogram_distance2(shuffleHistogram, GrayscaleHistogram)
print(distance2)


# <Exercise 2.5 (e)>
# RGB image.
# ranges = (0, 256) # the upper boundary is exclusive
# https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
bgr_planes = cv2.split(image)
imCh0 = cv2.calcHist(images=bgr_planes, channels=[0], mask=None, histSize=[256], ranges=[0, 256])
imCh1 = cv2.calcHist(images=bgr_planes, channels=[1], mask=None, histSize=[256], ranges=[0, 256])
imCh2 = cv2.calcHist(images=bgr_planes, channels=[2], mask=None, histSize=[256], ranges=[0, 256])

# imCh = cv2.calcHist(images=[image[0], image[1], image[2]], channels=[0, 1, 2], mask=None, histSize=[256, 256, 256], ranges=[0, 255, 0, 255, 0, 255])
# https://stackoverflow.com/questions/13730468/from-nd-to-1d-arrays
ravel0 = np.array(imCh0.ravel())
ravel1 = np.array(imCh1.ravel())
ravel2 = np.array(imCh2.ravel())

ravelArrays = np.array((ravel0, ravel1, ravel2))
ravelArraysT = ravelArrays.transpose()
# print("ravelArrays", ravelArrays.shape)
# print("ravelArraysT", ravelArraysT.shape)

show_histogram(ravelArraysT, 3, title="RGB Histogram")

# <Exercise 2.5 (f)>
# HSV image.
# TODO Why it is not appear as the same image in pdf exercise?
# since you just shuffle it in x axis
HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

bgr_planesHSV = cv2.split(HSV)
imChHSV0 = cv2.calcHist(images=bgr_planesHSV, channels=[0], mask=None, histSize=[256], ranges=[0, 256])
imChHSV1 = cv2.calcHist(images=bgr_planesHSV, channels=[1], mask=None, histSize=[256], ranges=[0, 256])
imChHSV2 = cv2.calcHist(images=bgr_planesHSV, channels=[2], mask=None, histSize=[256], ranges=[0, 256])

ravelHSV0 = np.array(imChHSV0.ravel())
ravelHSV1 = np.array(imChHSV1.ravel())
ravelHSV2 = np.array(imChHSV2.ravel())

HSVravelArrays = np.array((ravelHSV0, ravelHSV1, ravelHSV2))
HSVravelArraysT = HSVravelArrays.transpose()

show_histogram(HSVravelArraysT, 4, title="HSV Histogram")
####
# Since we shuffled grayscale before. Line 54
# np.random.shuffle() modify a sequence in-place by shuffling its contents.
# This function only shuffles the array along the first axis of a multi-dimensional array.
# The order of sub-arrays is changed but their contents remains the same.
# Note: This method changes the original list/tuple/string, it does not return a new list/tuple/string. It returns None.
grayscale2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show_image(grayscale2, 1, title="Grayscale Image", isGray=True)

show_image(cv2.cvtColor(grayscale, cv2.COLOR_BGR2RGB), 2, title="Shuffled Image", isGray=True)

imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
show_image(imageRGB, 3, title="RGB Image")

show_image(HSV, 4, title="HSV Image")

####
# Show the Matplotlib windows.
plt.show()

