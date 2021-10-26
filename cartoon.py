import time
from random import randint
import cv2
from matplotlib import pyplot as plt
import numpy as np

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def medianBlur(image):
    blur = cv2.medianBlur(image, 7)
    return blur

def laplacianFilter(image):
    laplacian = cv2.Laplacian(image, ddepth=-1, ksize=5, borderType=cv2.BORDER_DEFAULT)
    return laplacian

def threshold(image):
    coloredImage = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    ret, thresh = cv2.threshold(coloredImage,125, 255, type=cv2.THRESH_BINARY_INV)
    return thresh

def bilateral(image, thresholdImage):
    bilateral = cv2.bilateralFilter(image,9 , 9,7)
    for _  in range(6):
        # bilateral = cv2.bilateralFilter(bilateral,9 , 9,7)
        bilateral = cv2.bilateralFilter(bilateral, d=10, sigmaColor=250,sigmaSpace=250)
    cartoon = cv2.bitwise_and(bilateral, thresholdImage)   
    return cartoon

def kmean(image):
    total_color = 8
    k=total_color
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(image.shape)
    return result


def cartoon1(image):
    grayscaledImage = grayscale(image)
    blurredImage = medianBlur(grayscaledImage)
    laplacianImage = laplacianFilter(blurredImage)
    thresholdImage = threshold(laplacianImage)
    bilateralImage = bilateral(image,thresholdImage)
    return bilateralImage

def cartoon2(image):
    style = cv2.stylization(image, sigma_s=150, sigma_r=0.25)
    return style

def cartoon3(image):
    grayscaledImage = grayscale(image)
    blurredImage = medianBlur(grayscaledImage)
    laplacianImage = laplacianFilter(blurredImage)
    thresholdImage = threshold(laplacianImage)
    kmeans = kmean(image)
    bilateralImage = bilateral(kmeans,thresholdImage)
    return bilateralImage

files = ['img1.png','img2.jpg','img3.jpg','img4.jpg','img5.jpg','img6.jpg']
for filename in files:
    # filename = 'sherif_new.jpg'
    image = cv2.imread(filename)


    c1 = cartoon1(image)
    c2 = cartoon2(image)
    c3 = cartoon3(image)

    cv2.imwrite(f'output_cartoon1_{filename}',c1)
    cv2.imwrite(f'output_cartoon2_{filename}',c2)
    cv2.imwrite(f'output_cartoon4_{filename}',c3)
