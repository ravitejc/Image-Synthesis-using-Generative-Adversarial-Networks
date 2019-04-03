import os
from os.path import join
import argparse
import traceback
import pickle

import skipthoughts

import random
import numpy as np


def onehot_encoding_102_classes(path):
	allClasses = []
	oneHotEncoding = []
	try :
        file = open(path, 'r')
        #Read all 102 classes and remove newline character from end of line
		allClasses = file.readlines()
        for i, curClass in enumerate(allClasses):
            allclasses[i] = curClass.strip('\n')
        file.close()
	except IOError :
		print('Not able to open file containing information about 102 classes of flowers')
		traceback.print_stack()

	oneHotEncoding = np.zeros((102, 102))
	oneHotEncoding[np.arange(102), np.arange(102)] = 1

	return allClasses, oneHotEncoding, 102

def data_preprocessing(path):
    
    allimages = join(path, 'flowers/jpg')
    allcaptions = join(path, 'flowers/all_captions.txt')
    path = os.path.join(path, "flowers/allclasses.txt")
    classesWithCaptions = join(path, 'flowers/text_c10')

    images = []
    for image in os.listdir(allimages):
        if 'jpg' in image:
            images.append(image)

    allClasses, oneHotEncoding, totalClasses = onehot_encoding_102_classes(path)

    5CaptionsPerImageDict = {}
    imageOnehotEncodingDict = {}
    classFolders = []
    classNames = []
    imagesList = []

    for i in range(1, 103) :
        classFolder = join(classesWithCaptions, 'class_%.5d' % (i))
        classNames.append('class_%.5d' % (i))
        classFolders.append(classFolder)

        imageFiles = []
        for imageFile in os.listdir(classFolder):
            if 'txt' in imageFile:
                imageFiles.append(imageFile[0 :11] + ".jpg")

        for imageFile in imageFiles:
            imageOnehotEncodingDict[imageFile] = None

        for imageFile in imageFiles:
            5CaptionsPerImageDict[imageFile] = []

    for classFolder, className in zip(classFolders, classNames):
        captionFiles = []
        for captionFile in os.listdir(classFolder):
            if 'txt' in captionFile:
                captionFiles.append(captionFile)
        
        captionFilePath = join(classFolder, captionFile)
        for captionFile in caption_files:
            file = open(captionFilePath, 'r')
            captions = file.read()
            captions = captions.split('\n')

            file.close()

            imageFile = captionFile[0 :11] + ".jpg"

            imageOnehotEncodingDict[imageFile] = oneHotEncoding[allClasses.index(classname)]
            for caption in captions:
                if len(caption) > 0:
                    random.shuffle(caption)
                    5CaptionsPerImageDict[imageFile] += caption[0:5]

    #encoded captions using skipthoughts
    skipthought = skipthoughts.load_model()
    encodedCaptionsDict = {}
    for image in 5CaptionsPerImageDict:
        encodedCaptionsDict[image] = skipthoughts.encode(skipthought, 5CaptionsPerImageDict[image])

    imagesList = list(5CaptionsPerImageDict.keys())
    random.shuffle(imagesList)

    imagesForTraining = imagesList[0:int(len(imagesList)*0.9)]
    imagesForValidation = imagesList[int(len(imagesList)*0.9):-1]

    #Serialize files
    pickle.dump(5CaptionsPerImageDict, open(os.path.join(path, 'flowers', 'flowers_caps.pkl'), "wb"))
    pickle.dump(imagesForTraining, open(os.path.join(path, 'flowers', 'train_ids.pkl'), "wb"))
    pickle.dump(imagesForValidation, open(os.path.join(path, 'flowers', 'val_ids.pkl'), "wb"))
    pickle.dump(encodedCaptionsDict, open(join(path, 'flowers', 'flower_tv.pkl'), "wb"))
    pickle.dump(imageOnehotEncodingDict, open(join(path, 'flowers', 'flower_tc.pkl'), "wb"))

def main() :
    argParser = argparse.ArgumentargParser()
    argParser.add_argument('--path', type = str, default = 'Data', help = 'Directory for dataset')
    argParser.add_argument('--dataset', type=str, default='flowers', help='Dataset information')
    arguments = argParser.parse_arguments()

    if arguments.dataset == 'flowers':
        data_preprocessing(join(arguments.path, "datasets"))
    else:
        print('Dataset to preprocess is not found')


if __name__ == '__main__' :
    main()
