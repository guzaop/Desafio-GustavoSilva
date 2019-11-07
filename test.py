# Desafio - Gustavo Bandeira da Silva - Rede Neural Convolucional
import cv2
import glob
import numpy as np
import sys
import os
from model import Agent

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print ("Usage: python test.py <weights_filename> <test_image_directory>")
    else:
        weights_filename = sys.argv[1]
        test_image_directory = sys.argv[2]

        model = Agent()
        model.load(weights_filename)
        print("Model loaded.")
        files = (glob.glob(test_image_directory + "/*.bmp"))
        files = sorted(files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        for filename in files:
            img = cv2.imread(filename)
            img = cv2.resize(img, (246, 205), interpolation=cv2.INTER_AREA)
            img = img/255.0
            img = np.expand_dims(img, axis=0)
            prediction = model.make_prediction(img)
            print("Prediction for image '%s': " % (filename.split('/')[-1]), end='')
            if(prediction < 0.5):
                print("Sem anel" % (prediction))
            else:
                print("Com anel" % (prediction))
