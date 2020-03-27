import numpy as np; import pandas as pd
import cv2 as opencv
import urllib

print(opencv.__version__)

def test_numpy_matrix():
    newlst = [1,2,3]
    newarr = np.array(newlst)
    g = np.zeros(shape=(3,2))
    g = np.ones((2,4))
    print(g)

def show_panda():
    src = 'C:/Users/jedel/Pictures/Saved Pictures/1280px-Panda_Cub_from_Wolong,_Sichuan,_China.png'
    panda_img = opencv.imread(src)
    print(panda_img)
    panda_gray_img = opencv.cvtColor(panda_img,
                     opencv.COLOR_BGR2GRAY)
    opencv.imshow("Gray panda", panda_gray_img)
    opencv.imshow("Color panda", panda_gray_img)
    opencv.imwrite("gray_panda.png", panda_gray_img)
    opencv.waitKey(0)
    opencv.destroyAllWindows()

show_panda()