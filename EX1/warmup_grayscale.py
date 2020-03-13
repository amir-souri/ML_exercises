# Course: Introduction to Image Analysis and Machine Learning (ITU)
# Version: 2020.1

import cv2

# Full path where is located the input image.
filepath = "./inputs/lena.jpg"

# Open the image as a grayscale image.
image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

# Show the input image in a OpenCV window.
cv2.imshow("Image", image)

#import matplotlib.pyplot as plt
#plt use diffrent map function. To fix it add cmap= 'gray'. plt.imshow(image, cmap= 'gray')
#plt.imshow(image, cmap= 'gray')
#plt.show()


cv2.waitKey(0)

# Save the converted image.
# There should be a folder whos name is outputs. Otherwise it wont create 
# outputs folder
cv2.imwrite("./outputs/lena_grayscale.jpg", image)

# When everything done, release the OpenCV window.
cv2.destroyAllWindows()
