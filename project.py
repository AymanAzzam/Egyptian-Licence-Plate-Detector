import cv2 as cv
import numpy as np
from numba import jit
from commonfunctions import *
from tkinter import *
import os


dataBase = []
dataBase_b = []
database_characters = []
smalls = []
videoPlateTexts = {}
WIDTH_J = 60
HEIGHT_J = 60
WIDTH = 120
HEIGHT = 120


class character_j:
    def __init__(self, char, template = '', img = None):
        self.char = char
        if img is None:
            self.template = cv.imread(template, 0)
        else:
            self.template = img
        self.col_sum = np.zeros(shape=(HEIGHT_J,WIDTH_J))
        self.corr = 0


class character:
    def __init__(self, char, template = '', img = None):
        self.char = char
        if img is None:
            self.template = cv.imread(template, 0)
#             self.template=cv.adaptiveThreshold(self.template, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 75, 13);
        else:
            self.template = img
#             self.template=cv.adaptiveThreshold(self.template, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 75, 13);
        self.col_sum = np.zeros(shape=(HEIGHT_J,WIDTH_J))
        self.corr = 0


@jit(nopython=True)
def get_white_blue(image, aux):    
    h = image.shape[0]; w = image.shape[1]
    white = 200; local_thre = 30; global_thre = 60

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            b,g,r = image[y,x]
            
            s,m,l = np.sort(image[y,x])
            
            local_dis = (m-l)*(m-l)+(m-s)*(m-s)
            aux[y, x, 0] = 1 if (local_dis<local_thre*local_thre and abs(white-(s+m+l)/3)<global_thre) else 0
            aux[y, x, 1] = 1 if (s==r and l==b and r<100 and b>120 and b-g>20 and b-g<110) else 0


def check_blue(pixel):
        b,g,r = pixel
        s,m,l = np.sort(pixel)
        if (s==r and l==b and r<100 and b>120 and b-g>20 and b-g<110):
            return True
        else:
            return False


def sum_range(aux, Xmin, Ymin, Xmax, Ymax): 
    res = aux[Ymax][Xmax] 
    if (Ymin > 0): res = res - aux[Ymin - 1][Xmax]         
    if (Xmin > 0): res = res - aux[Ymax][Xmin - 1] 
    if (Ymin > 0 and Xmin > 0): res = res + aux[Ymin - 1][Xmin - 1] 
    return res 


def remove_noise(img):
    blur = cv.GaussianBlur(img, (3, 3), 0)
    return blur


#working well with bimodal images (Fast Algo)
def binarization_otsu(gray_img):
    ret,bin_img = cv.threshold(gray_img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return ret,bin_img


def detect_edges(gray_img): #Sobel Edge detection
    scale = 1; delta = 0; ddepth = cv.CV_16S
    grad_x = cv.Sobel(gray_img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray_img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x); abs_grad_y = cv.convertScaleAbs(grad_y)
    return cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


#depend on aspect ratio, center point of plate, plate color, far or not
def plate_criteria(cum_white, cum_blue, x, y, w, h, aspect_min, aspect_max, far): 
    area = w*h
    [Xmin,Ymin,Xmax,Ymax] = [x,y,x+w-1,y+h-1]
    if(h>0 and aspect_min < float(w)/h and float(w)/h < aspect_max): #Check Aspect ration
        if(area >= cum_white.shape[0] * cum_white.shape[1] * far): #check far or not
            white_ratio = sum_range(cum_white, Xmin, Ymin, Xmax, Ymax)/area*100
            blue_ratio = sum_range(cum_blue, Xmin, Ymin, Xmax, Ymax)/area*100
            if(white_ratio > 35 and white_ratio < 90 and blue_ratio > 7 and blue_ratio < 40):
                return True
    return False


def plate_contour(img, bin_img, aspect_min, aspect_max, far): #Image should be BGR Image not RGB
    #Because Some version return 2 parameters and other return 3 parameters
    major = cv.__version__.split('.')[0]
    if major == '3': img2, bounding_boxes, hierarchy= cv.findContours(bin_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    else: bounding_boxes, hierarchy= cv.findContours(bin_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    aux = np.copy(img); get_white_blue(img,aux)

    cum_white = np.cumsum(aux[:,:,0], axis = 0) ;cum_white = np.cumsum(cum_white, axis = 1).astype(np.int64) #To Avoid overflow in sum_range
    cum_blue = np.cumsum(aux[:,:,1], axis = 0); cum_blue = np.cumsum(cum_blue, axis = 1).astype(np.int64)   #To Avoid overflow in sum_range

    for box in bounding_boxes:    
        [x,y, w, h] = cv.boundingRect(box)
        if(plate_criteria(cum_white, cum_blue, x, y, w, h,aspect_min, aspect_max, far)):
            if(y-h/4>=0):
                return np.copy(img[y-int(h/4):y+h-1,x:x+w-1]),1
            else:    
                return np.copy(img[y:y+h-1,x:x+w-1]),1
    return img,0


def resize_image(img):
    if(img.shape[0]*img.shape[1]>1000000):
        h =  np.sqrt(1000000/(img.shape[1]*img.shape[0]))
        y = int(h*img.shape[0])
        x = int(h*img.shape[1])
        img = cv.resize(img, (x,y), interpolation = cv.INTER_AREA)
    return img


def rotate_blue(img):
    h,w,_ = img.shape
    x1 = int(img.shape[0]/4)
    y1 = 0
    while(y1 < h and not check_blue(img[y1][x1])):
        y1+=1
    x2 = img.shape[0]-int(img.shape[0]/4)
    y2 = 0
    while(y2 < h and not check_blue(img[y2][x2])):
        y2+=1
    center = (int(w/2),int(h/2))
    angle = np.arctan((y2-y1)/(x2-x1))*180*7/22
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, rotation_matrix, (w, h),flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return rotated


def crop_up(img):
    y = img.shape[0]
    x = int (img.shape[0]/2)
    for i in range(0,y):
        if(check_blue(img[i][x])):
            return img[i:y,0:img.shape[1]]
    return img


def localization(img): #take BGR image and return BGR image
    img = resize_image(img)
    img = remove_noise(img)
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    edges = detect_edges(gray_img)
  
    kernel = np.ones((5,5),np.uint8)
    closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    ret,bin_img = binarization_otsu(closing)

    plate_area_img,flag = plate_contour(img, bin_img, 1.4, 2.5, 0.01) 

    plate_area_img_bin = cv.adaptiveThreshold(cv.cvtColor(255-plate_area_img,cv.COLOR_BGR2GRAY),255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,8)

    plate_img,flag2 = plate_contour(plate_area_img, plate_area_img_bin, 1, 2.1, 0.1) 
    cropped = np.copy(plate_img)
    if(flag):
        #rotated = rotate_blue(plate_img)
        cropped = crop_up(plate_img)

    #plt.imshow(closing,cmap = 'gray')
    #plt.show()

    #plt.imshow(bin_img,cmap = 'gray')
    #plt.show()

    #plt.imshow(cv.cvtColor(plate_area_img,cv.COLOR_BGR2RGB))
    #plt.show()

    #plt.imshow(plate_area_img_bin,cmap = 'gray')
    #plt.show()

    #plt.imshow(cv.cvtColor(plate_img,cv.COLOR_BGR2RGB))
    #plt.show()
    
    return cropped,flag


def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0

	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True

	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1

	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))

	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


@jit(nopython = True)
def char_calculations_j(A, WIDTH_J, HEIGHT_J):
    A_mean = A.mean()
    col_A = 0
    corr_A = 0
    sum_list = np.zeros(shape=(HEIGHT_J,WIDTH_J))
    img_row = 0
    while img_row < HEIGHT_J:
        img_col = 0
        while img_col < WIDTH_J:
            col_A += (A[img_row, img_col] - A_mean) ** 2
            sum_list[img_row][img_col] = abs(A[img_row, img_col] - A_mean)
            img_col = img_col + 1
        corr_A += col_A
        col_A = 0
        img_row = img_row + 1  
    return corr_A,sum_list  


@jit(nopython = True)
def cal_corr_j(corr_A, corr_B, A_sum, B_sum):
    corr_both = np.multiply(A_sum, B_sum)
    corr_both = corr_both.sum()
    r = corr_both / math.sqrt(corr_A * corr_B)
    return r


@jit(nopython = True)
def my_resize_j(img, w, h):
    new_img = np.zeros(shape=(w, h))
    WIDTH_J,HEIGHT_J = img.shape
    Xwmin = 0
    Ywmin = 0
    Xwmax = WIDTH_J - 1
    Ywmax = HEIGHT_J - 1
    Xvmin = 0
    Yvmin = 0
    Xvmax = w - 1
    Yvmax = h - 1
    Sx = (Xvmax - Xvmin)/(Xwmax - Xwmin)
    Sy = (Yvmax - Yvmin)/(Ywmax - Ywmin)
    for i in range(HEIGHT_J):
        new_i = int(Yvmin + (i - Ywmin) * Sy)
        for j in range(WIDTH_J):
            new_j = int(Xvmin + (j - Xwmin) * Sx)
            new_img[new_j][new_i] = img[j][i]
            new_img[new_j][new_i] = img[j][i]
            new_img[new_j][new_i] = img[j][i]
    return new_img    


def buildDB():
	hamza = character_j('hamza','Recognition_Pictures/hamza1.jpg')
	no2taB = character_j('no2taB','Recognition_Pictures/no2ta1noon.jpg')
	no2taG = character_j('no2taG','Recognition_Pictures/no2ta6gem.jpg')

	hamza.template = my_resize_j(hamza.template,WIDTH_J,HEIGHT_J)
	no2taB.template = my_resize_j(no2taB.template,WIDTH_J,HEIGHT_J)
	no2taG.template = my_resize_j(no2taG.template,WIDTH_J,HEIGHT_J)
    
	hamza.corr, hamza.col_sum = char_calculations_j(hamza.template,HEIGHT_J,WIDTH_J)
	no2taB.corr, no2taB.col_sum = char_calculations_j(no2taB.template,HEIGHT_J,WIDTH_J)
	no2taG.corr, no2taG.col_sum = char_calculations_j(no2taG.template,HEIGHT_J,WIDTH_J)

	dataBase.append(hamza)
	dataBase.append(no2taB)
	dataBase.append(no2taG)


def isMiniLiter(imgI):
    letter = character_j('unk',img = imgI)
    letter.template = my_resize_j(letter.template,WIDTH_J,HEIGHT_J)
    letter.corr, letter.col_sum = char_calculations_j(letter.template,HEIGHT_J,WIDTH_J)
    for l in dataBase:
        temp1 = letter.template.astype(np.float32)
        temp2 = l.template.astype(np.float32)
        hist1=0
        hist2=0
        hist1=cv.calcHist([temp1],[0],None,[256],[0,256]) 
        hist2=cv.calcHist([temp2],[0],None,[256],[0,256]) 
        r =4
        r = cv.compareHist(hist1, hist2, method = cv.HISTCMP_CORREL)
        rCorr = cal_corr_j(letter.corr,l.corr,letter.col_sum,l.col_sum)
        if(rCorr>.75 and r > .5):
            return True
        return False


def buildDB_b():
    path = 'database/bar/'
    for filename in os.listdir(path):
        bar = character_j(filename.split('.')[0], path + filename)
        bar.template = my_resize_j(bar.template,WIDTH_J,HEIGHT_J)
        bar.corr, bar.col_sum = char_calculations_j(bar .template,HEIGHT_J,WIDTH_J)
        dataBase_b.append(bar)
	
    path = 'database/nesr/'
    for filename in os.listdir(path):
        nesr = character_j(filename.split('.')[0], path + filename)
        nesr.template = my_resize_j(nesr.template,WIDTH_J,HEIGHT_J)
        nesr.corr, nesr.col_sum = char_calculations_j(nesr .template,HEIGHT_J,WIDTH_J)
        dataBase_b.append(nesr)	


def isBar(imgI):
    letter = character_j('unk',img = imgI)
    letter.template = my_resize_j(letter.template,WIDTH_J,HEIGHT_J)
    letter.corr, letter.col_sum =char_calculations_j(letter.template,HEIGHT_J,WIDTH_J)
    for l in dataBase_b:
        temp1 = letter.template.astype(np.float32)
        temp2 = l.template.astype(np.float32)
        hist1=0
        hist2=0
        hist1=cv.calcHist([temp1],[0],None,[256],[0,256]) 
        hist2=cv.calcHist([temp2],[0],None,[256],[0,256]) 
        r = cv.compareHist(hist1, hist2, method = cv.HISTCMP_CORREL)
        rCorr = cal_corr_j(letter.corr,l.corr,letter.col_sum,l.col_sum)
        if(rCorr>.8 and r > .8):
            return True
    return False


@jit(nopython = True)
def within(a,start,length):
    if(a>=start and a <= start+length):
        return True
    return False


@jit(nopython = True)
def intersection(a,b):
    a = list(a)
    b = list(b)
    var = 16
    a[0]-=var
    a[1]-=var
    a[2]+=var
    a[3]+=var
    b[0]-=var
    b[1]-=var
    b[2]+=var
    b[3]+=var
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return False # or (0,0,0,0) ?
    return True


def get_chars_images_from_plate_image(imgI):
    dim = (1404, 746)
    imgI = cv.resize(imgI, dim, interpolation = cv.INTER_AREA)
    w = imgI.shape[0]
    h = imgI.shape[1]
    imgB = cv.blur(imgI,(10,10))
    imgB = cv.blur(imgB,(10,10))
    imgM= cv.medianBlur(imgB,5)
    imgO = cv.adaptiveThreshold(imgM, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 75, 13);
    k1=cv.getStructuringElement(cv.MORPH_RECT,(10,15))
    k2=cv.getStructuringElement(cv.MORPH_RECT,(5,5))

    c =  cv.morphologyEx(imgO, cv.MORPH_CLOSE, k2)
    o =  cv.morphologyEx(c, cv.MORPH_OPEN, k1)
    contours, hierarchy = cv.findContours(o, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sort_contours(contours)[0]
    d = np.ones_like(imgI)


    imgs = []
    conts = []
    rects = []
    mendol = []
    mendolc = []
    T = False
    for contour in contours:
        x,y,w,h = rect = cv.boundingRect(contour)
        if(y>300 and x > 50 and y+h<750 and cv.contourArea(contour) > 29*29 and y < 575 and cv.contourArea(contour) <100000):#and cv.contourArea(contour)>2000):
            mendolc.append(contour)
            for i,r in enumerate(rects):
                if(x-20<r[0] and x+w+20 > r[0]+r[2])or (x>r[0]-20 and x+w-20 < r[0]+r[2]):
                    T = True
                    miniImg = np.copy(imgI[y:y+h,x:x+h])
                    if miniImg is not None:
                        if(isMiniLiter(miniImg)):
                            minY = min(y,r[1])                    
                            maxH = max(y+h,r[1]+r[3])-minY
                            minX = min(x,r[0])                    
                            maxW = max(x+w,r[0]+r[2])-minX
                            rects[i] = (minX,minY,maxW,maxH)
                            mendolc.append(contour)
                            break
                if (intersection(rect,r)):
                    T = True
                    minY = min(y,r[1])                    
                    maxH = max(y+h,r[1]+r[3])-minY
                    minX = min(x,r[0])                    
                    maxW = max(x+w,r[0]+r[2])-minX
                    rects[i] = (minX,minY,maxW,maxH)
                    mendolc.append(contour)
                    break
            if(T):
                smalls.append(rect)
                T = False
                continue
            rects.append(rect)
            mendol.append(rect)
            conts.append(contour)
    for rect in rects:
        imgX = None
        imgX = np.copy(imgI[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]);
        if imgX is not None:
            if(not isBar(imgX)):
                imgs.append(imgX)
    return imgs


@jit(nopython=True,parallel=True)
def char_calculations(A, WIDTH, HEIGHT):
    A_mean = A.mean()
    col_A = 0
    corr_A = 0
    sum_list = np.zeros(shape=(HEIGHT,WIDTH))
    img_row = 0
    while img_row < HEIGHT:
        img_col = 0
        while img_col < WIDTH:
            col_A += (A[img_row, img_col] - A_mean) ** 2
            sum_list[img_row][img_col] = abs(A[img_row, img_col] - A_mean)
            img_col = img_col + 1
        corr_A += col_A
        col_A = 0
        img_row = img_row + 1  
    return corr_A,sum_list  


def buildDB_D():
	Alf = character("alf", 'Final_All_pgm_charachters_inNumbersSequence_new/alf3.jpg')
	Alf2 = character("alf", 'Final_All_pgm_charachters_inNumbersSequence_new/alf5.png')
	Sen = character("sen", 'Final_All_pgm_charachters_inNumbersSequence_new/sen.jpg')
	Non = character("non", 'Final_All_pgm_charachters_inNumbersSequence_new/non2.png')
	Non2 = character("non", 'Final_All_pgm_charachters_inNumbersSequence_new/non5.png')
	Yeh = character("yeh", 'Final_All_pgm_charachters_inNumbersSequence_new/yeh.jpg')
	Lam = character("lam", 'Final_All_pgm_charachters_inNumbersSequence_new/lam3.jpg')
	Lam2 = character("lam", 'Final_All_pgm_charachters_inNumbersSequence_new/lam.png')
	Bih = character("bih", 'Final_All_pgm_charachters_inNumbersSequence_new/30.jpg')
	Dal = character("dal", 'Final_All_pgm_charachters_inNumbersSequence_new/32.jpg')
	Dal2 = character("dal", 'Final_All_pgm_charachters_inNumbersSequence_new/dal4.jpg')
	Reh = character("reh", 'Final_All_pgm_charachters_inNumbersSequence_new/36.jpg')
	Reh2 = character("reh", 'Final_All_pgm_charachters_inNumbersSequence_new/reh5.png')
	Kaf = character("kaf", 'Final_All_pgm_charachters_inNumbersSequence_new/86.jpg')
	Kaf2 = character("kaf", 'Final_All_pgm_charachters_inNumbersSequence_new/88.jpg')
	Mim = character("mim", 'Final_All_pgm_charachters_inNumbersSequence_new/33.jpg')
	Waw = character("waw", 'Final_All_pgm_charachters_inNumbersSequence_new/7.jpg')
	Waw2 = character("waw", 'Final_All_pgm_charachters_inNumbersSequence_new/waw2.jpg')
	Tah = character("tah", 'Final_All_pgm_charachters_inNumbersSequence_new/82.jpg')
	Sad = character("sad", 'Final_All_pgm_charachters_inNumbersSequence_new/42.jpg')
	Gem = character("gem", 'Final_All_pgm_charachters_inNumbersSequence_new/102.jpg')
	Ein = character("ein", 'Final_All_pgm_charachters_inNumbersSequence_new/ein.png')
	Heh = character("heh", 'Final_All_pgm_charachters_inNumbersSequence_new/heh2.jpg')
	Heh2 = character("heh", 'Final_All_pgm_charachters_inNumbersSequence_new/heh3.png')
	Heh3 = character("heh", 'Final_All_pgm_charachters_inNumbersSequence_new/heh4.png')
	Fih = character("Fih", 'Final_All_pgm_charachters_inNumbersSequence_new/fih3.png')
	Fih2 = character("Fih", 'Final_All_pgm_charachters_inNumbersSequence_new/fih2.jpg')
	Yeh = character("yeh", 'Final_All_pgm_charachters_inNumbersSequence_new/yeh.jpg')

	dim = (WIDTH,HEIGHT)
	Alf.template = cv.resize(Alf.template, dim, interpolation = cv.INTER_AREA)
	Alf2.template = cv.resize(Alf2.template, dim, interpolation = cv.INTER_AREA)
	Sen.template = cv.resize(Sen.template , dim, interpolation = cv.INTER_AREA)
	Non.template = cv.resize(Non.template , dim, interpolation = cv.INTER_AREA)
	Non2.template =cv.resize(Non2.template, dim, interpolation = cv.INTER_AREA)
	Yeh.template = cv.resize(Yeh.template , dim, interpolation = cv.INTER_AREA)
	Lam.template = cv.resize(Lam.template , dim, interpolation = cv.INTER_AREA)
	Lam2.template =cv.resize(Lam2.template, dim, interpolation = cv.INTER_AREA)
	Bih.template = cv.resize(Bih.template , dim, interpolation = cv.INTER_AREA)
	Dal.template = cv.resize(Dal.template , dim, interpolation = cv.INTER_AREA)
	Dal2.template =cv.resize(Dal2.template, dim, interpolation = cv.INTER_AREA)
	Reh.template = cv.resize(Reh.template , dim, interpolation = cv.INTER_AREA)
	Reh2.template =cv.resize(Reh2.template, dim, interpolation = cv.INTER_AREA)
	Kaf.template = cv.resize(Kaf.template , dim, interpolation = cv.INTER_AREA)
	Kaf2.template =cv.resize(Kaf2.template, dim, interpolation = cv.INTER_AREA)
	Mim.template = cv.resize(Mim.template , dim, interpolation = cv.INTER_AREA)
	Waw.template = cv.resize(Waw.template , dim, interpolation = cv.INTER_AREA)
	Waw2.template =cv.resize(Waw2.template, dim, interpolation = cv.INTER_AREA)
	Tah.template = cv.resize(Tah.template , dim, interpolation = cv.INTER_AREA)
	Sad.template = cv.resize(Sad.template , dim, interpolation = cv.INTER_AREA)
	Gem.template = cv.resize(Gem.template , dim, interpolation = cv.INTER_AREA)
	Ein.template = cv.resize(Ein.template , dim, interpolation = cv.INTER_AREA)
	Heh.template = cv.resize(Heh.template , dim, interpolation = cv.INTER_AREA)
	Heh2.template =cv.resize(Heh2.template, dim, interpolation = cv.INTER_AREA)
	Heh3.template =cv.resize(Heh3.template, dim, interpolation = cv.INTER_AREA)
	Fih.template = cv.resize(Fih.template , dim, interpolation = cv.INTER_AREA)
	Fih2.template =cv.resize(Fih2.template, dim, interpolation = cv.INTER_AREA)
	Yeh.template = cv.resize(Yeh.template , dim, interpolation = cv.INTER_AREA)


	Alf.corr, Alf.col_sum = char_calculations(Alf.template, HEIGHT, WIDTH)
	Alf2.corr, Alf2.col_sum = char_calculations(Alf2.template, HEIGHT, WIDTH)
	Sen.corr, Sen.col_sum = char_calculations(Sen.template, HEIGHT, WIDTH)
	Non.corr, Non.col_sum = char_calculations(Non.template, HEIGHT, WIDTH)
	Non2.corr, Non2.col_sum = char_calculations(Non2.template, HEIGHT, WIDTH)
	Yeh.corr, Yeh.col_sum = char_calculations(Yeh.template, HEIGHT, WIDTH)
	Lam.corr, Lam.col_sum = char_calculations(Lam.template, HEIGHT, WIDTH)
	Lam2.corr, Lam2.col_sum = char_calculations(Lam2.template, HEIGHT, WIDTH)
	Bih.corr, Bih.col_sum = char_calculations(Bih.template, HEIGHT, WIDTH)
	Dal.corr, Dal.col_sum = char_calculations(Dal.template, HEIGHT, WIDTH)
	Dal2.corr, Dal2.col_sum = char_calculations(Dal2.template, HEIGHT, WIDTH)
	Reh.corr, Reh.col_sum = char_calculations(Reh.template, HEIGHT, WIDTH)
	Reh2.corr, Reh2.col_sum = char_calculations(Reh2.template, HEIGHT, WIDTH)
	Kaf.corr, Kaf.col_sum = char_calculations(Kaf.template, HEIGHT, WIDTH)
	Kaf2.corr, Kaf2.col_sum = char_calculations(Kaf2.template, HEIGHT, WIDTH)
	Mim.corr, Mim.col_sum = char_calculations(Mim.template, HEIGHT, WIDTH)
	Waw.corr, Waw.col_sum = char_calculations(Waw.template, HEIGHT, WIDTH)
	Waw2.corr, Waw2.col_sum = char_calculations(Waw2.template, HEIGHT, WIDTH)
	Tah.corr, Tah.col_sum = char_calculations(Tah.template, HEIGHT, WIDTH)
	Sad.corr, Sad.col_sum = char_calculations(Sad.template, HEIGHT, WIDTH)
	Gem.corr, Gem.col_sum = char_calculations(Gem.template, HEIGHT, WIDTH)
	Ein.corr, Ein.col_sum = char_calculations(Ein.template, HEIGHT, WIDTH)
	Heh.corr, Heh.col_sum = char_calculations(Heh.template, HEIGHT, WIDTH)
	Heh2.corr, Heh2.col_sum = char_calculations(Heh2.template, HEIGHT, WIDTH)
	Heh3.corr, Heh3.col_sum = char_calculations(Heh3.template, HEIGHT, WIDTH)
	Fih.corr, Fih.col_sum = char_calculations(Fih.template, HEIGHT, WIDTH)
	Fih2.corr, Fih2.col_sum = char_calculations(Fih2.template, HEIGHT, WIDTH)
	Yeh.corr, Yeh.col_sum = char_calculations(Yeh.template, HEIGHT, WIDTH)


	# Numbers
	One = character("1", 'Final_All_pgm_charachters_inNumbersSequence_new/3.jpg')
	Two = character("2", 'Final_All_pgm_charachters_inNumbersSequence_new/4.jpg')
	Three = character("3", 'Final_All_pgm_charachters_inNumbersSequence_new/8.jpg')
	Four = character("4", 'Final_All_pgm_charachters_inNumbersSequence_new/11.jpg')
	Five = character("5", 'Final_All_pgm_charachters_inNumbersSequence_new/15.jpg')
	Six = character("6", 'Final_All_pgm_charachters_inNumbersSequence_new/18.jpg')
	Seven = character("7", 'Final_All_pgm_charachters_inNumbersSequence_new/21.jpg')
	Eight = character("8", 'Final_All_pgm_charachters_inNumbersSequence_new/25.jpg')
	Nine = character("9", 'Final_All_pgm_charachters_inNumbersSequence_new/27.jpg')


	One.template =   cv.resize(One.template , dim, interpolation = cv.INTER_AREA)
	Two.template =   cv.resize(Two.template , dim, interpolation = cv.INTER_AREA)
	Three.template = cv.resize(Three.template , dim, interpolation = cv.INTER_AREA)
	Four.template =  cv.resize(Four.template , dim, interpolation = cv.INTER_AREA)
	Five.template =  cv.resize(Five.template , dim, interpolation = cv.INTER_AREA)
	Six.template =   cv.resize(Six.template , dim, interpolation = cv.INTER_AREA)
	Seven.template = cv.resize(Seven.template , dim, interpolation = cv.INTER_AREA)
	Eight.template = cv.resize(Eight.template , dim, interpolation = cv.INTER_AREA)
	Nine.template =  cv.resize(Nine.template , dim, interpolation = cv.INTER_AREA)


	One.corr, One.col_sum = char_calculations(One.template, HEIGHT, WIDTH)
	Two.corr, Two.col_sum = char_calculations(Two.template, HEIGHT, WIDTH)
	Three.corr, Three.col_sum = char_calculations(Three.template, HEIGHT, WIDTH)
	Four.corr, Four.col_sum = char_calculations(Four.template, HEIGHT, WIDTH)
	Five.corr, Five.col_sum = char_calculations(Five.template, HEIGHT, WIDTH)
	Six.corr, Six.col_sum = char_calculations(Six.template, HEIGHT, WIDTH)
	Seven.corr, Seven.col_sum = char_calculations(Seven.template, HEIGHT, WIDTH)
	Eight.corr, Eight.col_sum = char_calculations(Eight.template, HEIGHT, WIDTH)
	Nine.corr, Nine.col_sum = char_calculations(Nine.template, HEIGHT, WIDTH)

	# Add to database
	database_characters.append(Alf)
	database_characters.append(Alf2)
	database_characters.append(Bih)
	database_characters.append(Dal)
	database_characters.append(Dal2)
	database_characters.append(Reh)
	database_characters.append(Reh2)
	database_characters.append(Sen)
	database_characters.append(Kaf)
	database_characters.append(Kaf2)
	database_characters.append(Mim)
	database_characters.append(Tah)
	database_characters.append(Sad)
	database_characters.append(Waw)
	database_characters.append(Waw2)
	database_characters.append(Gem)
	database_characters.append(Lam)
	# database_characters.append(Lam2)
	database_characters.append(Yeh)
	database_characters.append(Non)
	database_characters.append(Non2)
	database_characters.append(Ein)
	database_characters.append(Heh)
	database_characters.append(Heh2)
	database_characters.append(Heh3)
	database_characters.append(Fih)
	database_characters.append(Fih2)
	database_characters.append(Yeh)


	database_characters.append(One)
	database_characters.append(Two)
	database_characters.append(Three)
	database_characters.append(Four)
	database_characters.append(Five)
	database_characters.append(Six)
	database_characters.append(Seven)
	database_characters.append(Eight)
	database_characters.append(Nine)


@jit(nopython=True,parallel=True)
def cal_corr(corr_A, corr_B, A_sum, B_sum):
    corr_both = np.multiply(A_sum, B_sum)
    corr_both = corr_both.sum()
    r = corr_both / math.sqrt(corr_A * corr_B)
    return r


def getSimilarity(img1, img2 ):
    dim = (120,120)
    img1 = cv.GaussianBlur(img1,(19,19),0)
    img2 = cv.GaussianBlur(img2,(19,19),0)
    img1 = cv.resize(img1, dim, interpolation = cv.INTER_AREA)
    img2 = cv.resize(img2, dim, interpolation = cv.INTER_AREA)
    ret2,img1 = cv.threshold(img1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret2,img2 = cv.threshold(img2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    sim = img1 - img2
    sim = sim * sim
    sim = np.sum(sim)
    sim = np.sqrt(sim)
    return sim


def get_image_char(image):
    unkChar = character('Unk', img = image)
    r = 500000000000
    curr_r = 500000000000
    for j in database_characters:
        curr_r = similarity = getSimilarity(unkChar.template,j.template)
        if curr_r < r:
            unkChar.char = j.char
            r = curr_r
    return unkChar.char


def get_plate_text_from_char_images(imgs):
    plate = ""
    for img in imgs:
        unkChar = get_image_char(img)
        plate = plate + unkChar + " "
    return plate


def vote_plate_text_for_video(plateText):
    if plateText in videoPlateTexts:
        videoPlateTexts[plateText]+=1
    else:
        videoPlateTexts[plateText]=1
    bestPlateText = "None found"
    maxOcc = 0
    for s in videoPlateTexts.keys():
        if videoPlateTexts[s]>maxOcc and s != "":
            bestPlateText = s
            maxOcc = videoPlateTexts[s]
    print(videoPlateTexts)
    print("Best Plate Text is " + bestPlateText)
    return bestPlateText


def get_plate_text_from_image(image):
    plateText = ""
    if image is not None:
        plate, flag = localization(image)
        if(flag):
            charsImages = get_chars_images_from_plate_image(cv.cvtColor(plate,cv.COLOR_BGR2GRAY))
            plateText = get_plate_text_from_char_images(charsImages)
            print("plateText is " + plateText)
        cv.imshow("Localization",plate)
    return plateText


def get_plate_text_from_video(videoCapture):
    bestPlateText = ""
    if (videoCapture.isOpened()== False): 
        print("Error opening video stream or file")
    while(videoCapture.isOpened()):
        #get the Frame from video
        _, img = videoCapture.read()
        if img is not None:
            plateText = get_plate_text_from_image(img)
            bestPlateText = vote_plate_text_for_video(plateText)
        else:
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    return bestPlateText


def process_video(fileName):
	videoCapture = cv.VideoCapture(fileName)
	get_plate_text_from_video(videoCapture) 
	videoCapture.release();
	cv.destroyAllWindows();
