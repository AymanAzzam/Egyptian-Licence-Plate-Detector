import cv2 as cv
import os
from consts import WIDTH_J, HEIGHT_J, WIDTH, HEIGHT
from commonfunctions import my_resize, char_calculations, character


def build_hamza_no2ta_db():
    hamzaNo2taDB = []

    path = 'database/hamza/'
    for filename in os.listdir(path):
        hamza = character(filename.split('.')[0], path + filename)
        hamza.template = my_resize(hamza.template,WIDTH_J,HEIGHT_J)
        hamza.corr, hamza.col_sum = char_calculations(hamza .template,HEIGHT_J,WIDTH_J)
        hamzaNo2taDB.append(hamza)
	
    path = 'database/no2ta/'
    for filename in os.listdir(path):
        no2ta = character(filename.split('.')[0], path + filename)
        no2ta.template = my_resize(no2ta.template,WIDTH_J,HEIGHT_J)
        no2ta.corr, no2ta.col_sum = char_calculations(no2ta .template,HEIGHT_J,WIDTH_J)
        hamzaNo2taDB.append(no2ta)

    return hamzaNo2taDB


def build_bar_nesr_db():
    barNesrDB = []

    path = 'database/bar/'
    for filename in os.listdir(path):
        bar = character(filename.split('.')[0], path + filename)
        bar.template = my_resize(bar.template,WIDTH_J,HEIGHT_J)
        bar.corr, bar.col_sum = char_calculations(bar .template,HEIGHT_J,WIDTH_J)
        barNesrDB.append(bar)
	
    path = 'database/nesr/'
    for filename in os.listdir(path):
        nesr = character(filename.split('.')[0], path + filename)
        nesr.template = my_resize(nesr.template,WIDTH_J,HEIGHT_J)
        nesr.corr, nesr.col_sum = char_calculations(nesr .template,HEIGHT_J,WIDTH_J)
        barNesrDB.append(nesr)	

    return barNesrDB


def build_characters_db():
    charactersDB = []

    path = 'database/characters/'
    for filename in os.listdir(path):
        charac = character(filename.split('.')[0], path + filename)
        dim = (WIDTH,HEIGHT)
        charac.template = cv.resize(charac.template, dim, interpolation = cv.INTER_AREA)
        charac.corr, charac.col_sum = char_calculations(charac .template,HEIGHT_J,WIDTH_J)
        charactersDB.append(charac)
    
    return charactersDB


def build_databases():
    hamzaNo2taDB = build_hamza_no2ta_db()
    barNesrDB = build_bar_nesr_db()
    charactersDB = build_characters_db()

    return hamzaNo2taDB, barNesrDB, charactersDB