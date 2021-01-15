import array
import OpenEXR
import Imath
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def exr2flow(exr, w,h):
  file = OpenEXR.InputFile(exr)

  # Compute the size
  dw = file.header()['dataWindow']
  sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

  FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
  (R,G,B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]

  img = np.zeros((h,w,3), np.float64)
  img[:,:,0] = np.array(R).reshape(img.shape[0],-1)
  img[:,:,1] = -np.array(G).reshape(img.shape[0],-1)

  hsv = np.zeros((h,w,3), np.uint8)
  hsv[...,1] = 255

  mag, ang = cv2.cartToPolar(img[...,0], img[...,1])
  hsv[...,0] = ang*180/np.pi/2
  hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
  bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

  return img, bgr, mag,ang
  
try: 
    os.mkdir("processed") 
except OSError as error: 
    print(error) 

for root, directories, files in os.walk(".", topdown=False):
    for file in files : 
        print(file)
        img,bgr,mag,ang = exr2flow(file,1920,1080)

        fig = plt.figure()
        plt.imshow(bgr)
        plt.colorbar()
        ##plt.show()
        plt.savefig("processed/"+file+".png")