

import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from numba import jit
import cv2 as cv
from consts import WIDTH_J, HEIGHT_J


class character:
    def __init__(self, char, template = '', img = None):
        self.char = char
        if img is None:
            self.template = cv.imread(template, 0)
        else:
            self.template = img
        self.col_sum = np.zeros(shape=(HEIGHT_J, WIDTH_J))
        self.corr = 0


# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


@jit(nopython = True)
def my_resize(img, w, h):
    new_img = np.zeros(shape=(w, h))
    width, height = img.shape
    Xwmin = 0
    Ywmin = 0
    Xwmax = width - 1
    Ywmax = height - 1
    Xvmin = 0
    Yvmin = 0
    Xvmax = w - 1
    Yvmax = h - 1
    Sx = (Xvmax - Xvmin)/(Xwmax - Xwmin)
    Sy = (Yvmax - Yvmin)/(Ywmax - Ywmin)
    for i in range(height):
        new_i = int(Yvmin + (i - Ywmin) * Sy)
        for j in range(width):
            new_j = int(Xvmin + (j - Xwmin) * Sx)
            new_img[new_j][new_i] = img[j][i]
            new_img[new_j][new_i] = img[j][i]
            new_img[new_j][new_i] = img[j][i]
    return new_img    


@jit(nopython=True,parallel=True)
def char_calculations(A, width, height):
    A_mean = A.mean()
    col_A = 0
    corr_A = 0
    sum_list = np.zeros(shape=(height,width))
    img_row = 0
    while img_row < height:
        img_col = 0
        while img_col < width:
            col_A += (A[img_row, img_col] - A_mean) ** 2
            sum_list[img_row][img_col] = abs(A[img_row, img_col] - A_mean)
            img_col = img_col + 1
        corr_A += col_A
        col_A = 0
        img_row = img_row + 1  
    return corr_A,sum_list  
