# Desafio - Gustavo Bandeira da Silva - Rede Neural Convolucional
import cv2
import glob
import numpy as np
import sys
import Augmentor
from model import Agent
import shutil
import os

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('''Usage: python train.py <com_anel_directory> <sem_anel_directory> [model_save_name.h5]''')
    else:
        com_anel_directory = sys.argv[1]
        sem_anel_directory = sys.argv[2]

        if os.path.exists(com_anel_directory + '/augmented') and os.path.isdir(com_anel_directory + '/augmented'):
            shutil.rmtree(com_anel_directory + '/augmented')

        if os.path.exists(sem_anel_directory + '/augmented') and os.path.isdir(sem_anel_directory + '/augmented'):
            shutil.rmtree(sem_anel_directory + '/augmented')

        print("Found %d items on directory '%s', starting data augmentation..." % (len(glob.glob(com_anel_directory + "/*.bmp")), com_anel_directory))
        p = Augmentor.Pipeline(com_anel_directory, output_directory="augmented")
        p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
        p.flip_left_right(probability=0.5)
        p.zoom_random(probability=0.5, percentage_area=0.95)
        p.resize(probability=1.0, width=246, height=205)
        p.sample(250)
        print("Data augmentation done on directory '%s'." % (com_anel_directory))

        print("Found %d items on directory '%s', starting data augmentation..." % (len(glob.glob(sem_anel_directory + "/*.bmp")), sem_anel_directory))
        p = Augmentor.Pipeline(sem_anel_directory, output_directory="augmented")
        p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
        p.flip_left_right(probability=0.5)
        p.zoom_random(probability=0.5, percentage_area=0.95)
        p.resize(probability=1.0, width=246, height=205)
        p.sample(250)
        print("Data augmentation done on directory '%s'." % (sem_anel_directory))

        model = Agent()

        imgs = []
        labels = []

        for filename in glob.glob(com_anel_directory + "/augmented/*.bmp"):
            img = cv2.imread(filename)
            img = img/255.0
            imgs.append(img)
            labels.append(1.0)

        for filename in glob.glob(sem_anel_directory + "/augmented/*.bmp"):
            img = cv2.imread(filename)
            img = img/255.0
            imgs.append(img)
            labels.append(0.0)

        imgs = np.array(imgs)
        labels = np.array(labels)

        print("Dataset ready for training. Initiating fit process...")

        model.train(imgs, labels)
        
        if len(sys.argv) > 3:
            save_name = sys.argv[3]
        else:
            save_name = "model.h5"

        model.save(save_name)

        print("Model trained and saved on file '%s'." % (save_name))
