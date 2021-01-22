#!/usr/bin/python                                                                                                             

'''
author: Tobias Weis
'''

import OpenEXR
import Imath
import array
import numpy as np
import csv
import time
import datetime
import h5py
import matplotlib.pyplot as plt
import os

def exr2numpy(exr, maxvalue=1.,normalize=True):
    """ converts 1-channel exr-data to 2D numpy arrays """                                                                    
    file = OpenEXR.InputFile(exr)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R") ]

    # create numpy 2D-array
    img = np.zeros((sz[1],sz[0],3), np.float64)

    # normalize
    data = np.array(R)
    data[data > maxvalue] = maxvalue

    if normalize:
        data /= np.max(data)

    img = np.array(data).reshape(img.shape[0],-1)

    return img


try: 
    os.mkdir("processed") 
except OSError as error: 
    print(error) 

for root, directories, files in os.walk(".", topdown=False):
    for file in files :
        print(file)
        depth_data = exr2numpy(file, maxvalue=15, normalize=False)

        fig = plt.figure()
        plt.imshow(depth_data)
        plt.colorbar()
       # plt.show()
        plt.savefig("processed/"+file+".png")