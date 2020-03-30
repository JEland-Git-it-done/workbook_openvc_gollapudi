import numpy as np; import pandas as pd; import random
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

def load_and_write_video():

    file = opencv.VideoCapture(r"C:/Users/jedel/Videos/Escape From Tarkov/Escape From Tarkov 2019.11.29 - 11.12.24.08.DVR.1575022790461.mp4")
    fcc = opencv.VideoWriter_fourcc(*"XVID")
    out = opencv.VideoWriter("new_tark.avi", fcc, 28, (640, 360))
    print("Working")
    while True:
        ret, f = file.read()
        f2 = opencv.flip(f, 1)
        if not ret:
            break
        opencv.imshow("frame2", f2)
        opencv.imshow("frame", f)
        print("Working in loop")
        out.write(f2)
        print("Writing")

        if opencv.waitKey(20) & 0xFF == ord('q'):
            print("Getting to key")
            break

    out.release()
    file.release()
    opencv.destroyAllWindows()

def manipulating_pixels():
    panda = opencv.imread(panda_src)
    pixel = panda[200,250]
    print(pixel)
    panda[200,250]=(255,0,0) #variable is assigning colour
    #Using RBG scale
    panda[200:250, 200:350] = (0,255,0)
    print(type(panda), "all fine")
    opencv.imshow("modified pixel", panda)
    opencv.waitKey(0)
    #could save output photo, code attached below
    opencv.imwrite("green_pixel_panda.png", panda)

def drawing_shapes():
    panda = opencv.imread(panda_src)
    opencv.line(panda, (25,21), (100, 100), (255,0,0), 5)
    opencv.rectangle(panda, (25,21), (200,200), (0, 255,0),2)
    opencv.circle(panda, (50,50), 50, (0,0,255), -1)

    opencv.imshow("Geomety", panda)
    opencv.waitKey(0)

def translate_image():
    panda = opencv.imread(panda_src)
    num_rows, num_cols = panda.shape[:2]
    trans_matrix = np.float32([ [1,0,70], [0,1,110] ])
    img_translation = opencv.warpAffine(panda, trans_matrix,
                                        (num_cols, num_rows))
    opencv.imshow("Translation", img_translation)
    opencv.waitKey()

def rotate_image():
    panda = opencv.imread(panda_src)
    angle = []
    for i in range(3):
        num = random.randint(0,360)
        angle.append(float(num))
    print(type(angle))
    num_rows, num_cols = panda.shape[:2]
    translation_matrix = np.float32(
        [[1,0,int(0.5*num_cols)],
        [0,1,int(0.5*num_rows)],
        [2 * num_cols, 2 * num_rows]]
    )
    rotation_matrix = opencv.getRotationMatrix2D((tuple(num_cols), tuple(num_rows)),
    img_translation = opencv.warpAffine(panda, translation_matrix, (1),
    img_rotation = opencv.warpAffine(img_translation, rotation_matrix,
                                     (2*num_cols, 2*num_rows))))
    opencv.imshow("rotation", img_rotation)
    opencv.waitKey(0)



rotate_image()