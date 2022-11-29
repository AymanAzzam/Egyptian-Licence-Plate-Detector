import cv2 as cv
import numpy as np
from numba import jit
from commonfunctions import *
from tkinter import *
import consts as c
import math
smalls = []


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
            if(white_ratio > c.PLATE_MIN_WHITE and white_ratio < c.PLATE_MAX_WHITE and blue_ratio > c.PLATE_MIN_BLUE and blue_ratio < c.PLATE_MAX_BLUE):
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

    plate_area_img,flag = plate_contour(img, bin_img, c.PLATE_MIN_ASPECT_FAR, c.PLATE_MAX_ASPECT_FAR, c.PLATE_FAR) 

    plate_area_img_bin = cv.adaptiveThreshold(cv.cvtColor(255-plate_area_img,cv.COLOR_BGR2GRAY),255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,8)

    plate_img,flag2 = plate_contour(plate_area_img, plate_area_img_bin, c.PLATE_MIN_ASPECT_NEAR, c.PLATE_MAX_ASPECT_NEAR, c.PLATE_NEAR) 
    cropped = np.copy(plate_img)
    if(flag):
        #rotated = rotate_blue(plate_img)
        cropped = crop_up(plate_img)

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


def isMiniLiter(imgI, hamzaNo2taDB):
    letter = character('unk',img = imgI)
    letter.template = my_resize(letter.template,c.WIDTH_J,c.HEIGHT_J)
    letter.corr, letter.col_sum = char_calculations(letter.template,c.HEIGHT_J,c.WIDTH_J)
    for l in hamzaNo2taDB:
        temp1 = letter.template.astype(np.float32)
        temp2 = l.template.astype(np.float32)
        hist1=0
        hist2=0
        hist1=cv.calcHist([temp1],[0],None,[256],[0,256]) 
        hist2=cv.calcHist([temp2],[0],None,[256],[0,256]) 
        r =4
        r = cv.compareHist(hist1, hist2, method = cv.HISTCMP_CORREL)
        rCorr = cal_corr(letter.corr,l.corr,letter.col_sum,l.col_sum)
        if(rCorr>.75 and r > .5):
            return True
        return False


def isBar(imgI, barNesrDB):
    letter = character('unk',img = imgI)
    letter.template = my_resize(letter.template,c.WIDTH_J,c.HEIGHT_J)
    letter.corr, letter.col_sum =char_calculations(letter.template,c.HEIGHT_J,c.WIDTH_J)
    for l in barNesrDB:
        temp1 = letter.template.astype(np.float32)
        temp2 = l.template.astype(np.float32)
        hist1=0
        hist2=0
        hist1=cv.calcHist([temp1],[0],None,[256],[0,256]) 
        hist2=cv.calcHist([temp2],[0],None,[256],[0,256]) 
        r = cv.compareHist(hist1, hist2, method = cv.HISTCMP_CORREL)
        rCorr = cal_corr(letter.corr,l.corr,letter.col_sum,l.col_sum)
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


def get_chars_images_from_plate_image(imgI, hamzaNo2taDB, barNesrDB):
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
                        if(isMiniLiter(miniImg, hamzaNo2taDB)):
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
            if(not isBar(imgX, barNesrDB)):
                imgs.append(imgX)
    return imgs


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


def get_image_char(image, charactersDB):
    unkChar = character('Unk', img = image)
    r = 500000000000
    curr_r = 500000000000
    for j in charactersDB:
        curr_r = similarity = getSimilarity(unkChar.template,j.template)
        if curr_r < r:
            unkChar.char = j.char
            r = curr_r
    return unkChar.char.split('_')[0]


def get_plate_text_from_char_images(imgs, charactersDB):
    plate = ""
    for img in imgs:
        unkChar = get_image_char(img, charactersDB)
        plate = plate + unkChar + " "
    return plate


def vote_plate_text_for_video(plateText, videoPlateTexts):
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


def get_plate_text_from_image(image, hamzaNo2taDB, barNesrDB, charactersDB):
    plateText = ""
    if image is not None:
        plate, flag = localization(image)
        if(flag):
            charsImages = get_chars_images_from_plate_image(cv.cvtColor(plate,cv.COLOR_BGR2GRAY), hamzaNo2taDB, barNesrDB)
            plateText = get_plate_text_from_char_images(charsImages, charactersDB)
            print("plateText is " + plateText)
        cv.imshow("Localization",plate)
    return plateText


def get_plate_text_from_video(videoCapture, hamzaNo2taDB, barNesrDB, charactersDB, videoPlateTexts):
    bestPlateText = ""
    if (videoCapture.isOpened()== False): 
        print("Error opening video stream or file")
    while(videoCapture.isOpened()):
        #get the Frame from video
        _, img = videoCapture.read()
        if img is not None:
            plateText = get_plate_text_from_image(img, hamzaNo2taDB, barNesrDB, charactersDB)
            bestPlateText = vote_plate_text_for_video(plateText, videoPlateTexts)
        else:
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    return bestPlateText


def process_video(fileName, hamzaNo2taDB, barNesrDB, charactersDB):
    videoPlateTexts = {}
    videoCapture = cv.VideoCapture(fileName)
    get_plate_text_from_video(videoCapture, hamzaNo2taDB, barNesrDB, charactersDB, videoPlateTexts) 
    videoCapture.release()
    cv.destroyAllWindows()
