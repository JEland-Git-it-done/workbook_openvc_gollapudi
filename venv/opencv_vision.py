import numpy as np; import pandas as pd;
import cv2 as opencv
import urllib; from matplotlib import pyplot as plt

print(opencv.__version__)
panda_src = 'C:/Users/jedel/Pictures/Saved Pictures/1280px-Panda_Cub_from_Wolong,_Sichuan,_China.png'
def test_numpy_matrix():
    newlst = [1,2,3]
    newarr = np.array(newlst)
    g = np.zeros(shape=(3,2))
    g = np.ones((2,4))
    print(g)

def show_panda():

    panda_img = opencv.imread(panda_src)
    print(panda_img)
    panda_gray_img = opencv.cvtColor(panda_img,
                     opencv.COLOR_BGR2GRAY)
    opencv.imshow("Gray panda", panda_gray_img)
    opencv.imshow("Color panda", panda_gray_img)
    opencv.imwrite("gray_panda.png", panda_gray_img)
    opencv.waitKey(0)
    opencv.destroyAllWindows()

def histogram_panda():
    src = panda_src
    panda_img = opencv.imread(src)
    histogram = opencv.calcHist([panda_img], [0],
                                None, [256], [0,256])
    plt.hist(histogram.ravel(), 256, [0,256])
    plt.show()
    color = ["b", "g", "r"]
    for i, col in enumerate(color):
        hist = opencv.calcHist([panda_img], [i],
                               None, [256], [0, 256])
        plt.plot(hist, color = col)
        plt.xlim([0,256]) #min-maximum x-value
    plt.show()

def webcam_test():
    cap = opencv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        opencv.imshow("frame", frame)
        key = opencv.waitKey(1)
        if key == 27:
            break

    cap.release()
    opencv.destroyAllWindows()

def load_video():
    file = r"C:\Users\jedel\Videos\Escape From Tarkov\Escape From Tarkov 2019.11.29 - 11.12.24.08.DVR.1575022790461.mp4"
    video = opencv.VideoCapture(file)
    while True:
        ret, frame = video.read()
        opencv.imshow("Escape From Tarkov 2019.11.29 - 11.12.24.08.DVR.1575022790461.mp4", frame)

        key = opencv.waitKey(25)
        if key == 27:
            break
    video.release()
    opencv.destroyAllWindows()
    print("ALL GOOD")
load_video()