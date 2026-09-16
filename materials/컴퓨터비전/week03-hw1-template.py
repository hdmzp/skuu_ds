import numpy as np
import cv2
import random

'''
Problem #1 Image Transformations
'''

# 1-1
def translate(image, x, y):
    # Write your code
    translation_matrix = None

    transformed_image = cv2.warpAffine()
    #################

    return transformed_image

def rotate(image, angle):
    # Write your code
    transform_matrix = None

    transformed_image = cv2.warpAffine()
    #################

    return transformed_image

def AffineTransformation(image, source_match_point, target_match_point):
    # Write your code
    transform_matrix = None

    transformed_image = cv2.warpAffine()
    #################

    return transformed_image

def PerspectiveTransformation(image, source_match_point, target_match_point):
    # Write your code
    transform_matrix = None

    transformed_image = cv2.warpPerspective()
    #################

    return transformed_image


'''
Problem #2 Linear Filters
'''

# 2-1

def Gaussian_filter(image):
    # Write your code
    # Set any parameters for the kernel

    result = cv2.GaussianBlur(...)

    return result

def Sobel(image):
    # Write your code
    # Set any parameters for the kernel
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.Sobel(...)

    return result

def Laplacian(image):
    # Write your code
    # Set any parameters for the kernel
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.Laplacian(...)

    return result

# 2-2

######## Don't modify this function ####################
def add_salt_pepper_noise(image, prob):
    # You can use prob argument as the probability of noise at each pixel
    result = np.zeros(image.shape, dtype=np.uint8)
    th = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rand = random.random()
            if rand < prob:
                result[i][j] = 0
            elif rand > th:
                result[i][j] = 255
            else:
                result[i][j] = image[i][j]
    return result
#########################################################


# Write your code
# GaussianBlur, Sobel, and Laplacian to the result image with noise


# 2-3
def median_blur(image):
    # Write your code
    # You can use cv2.medianBlur
    # Choose various sizes of the filter
    result = None

    return result

# Write your code
# Apply medianBlur to the result image with noise


...

'''
Problem #3 Image Pyramids
'''

# 3-1

# Use cv2.resize() and its argument "interpolation=..."
# Try different options of interpolation

# Write your code
# To down-sample an image, you should set the size smaller than the image
down_result_01 = cv2.resize(..., interpolation=...)
down_result_02 = cv2.resize(..., interpolation=...)
down_result_03 = cv2.resize(..., interpolation=...)

# Write your code
# To up-sample an image, you should set the size lager than the image
up_result_01 = cv2.resize(..., interpolation=...)
up_result_02 = cv2.resize(..., interpolation=...)
up_result_03 = cv2.resize(..., interpolation=...)

# 3-2

# Write your code
# Set the number of levels of pyramid as many as you want!
Gaussian_down_01 = cv2.pyrDown(...)
Gaussian_down_02 = cv2.pyrDown(...) # You should use Gaussian_down_01 as the input
... # You can repeat
Gaussian_up_01 = cv2.pyrUp(...)
Gaussian_up_02 = cv2.pyrUp(...)
Gaussian_up_03 = cv2.pyrUp(...)
...

# 3-3
## Restored by using Laplacian with "lenna.png"


# Write your code
Laplacian_down = None
Laplacian_up = None

# Restore image with Laplacian down and up
Restored_image = None 



'''
Problem #4 Median Blur
'''
def my_median_blur(image, size):
    # Write your code
    # You can use any library for sorting like numpy.sort()
    # You do not have to consider the issue of padding

    result = None

    return result
