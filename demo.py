import cv2
import numpy as np
import kNN
import pandas

global img, flag
global point1, point2
global points
def on_mouse(event, x, y, flags, param):
    global img, point1, point2, flag
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 5)
        cv2.imshow('image', img2)
        min_x = min(point1[0],point2[0])
        min_y = min(point1[1],point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        cut_img = img[min_y:min_y+height, min_x:min_x+width]
        cv2.imwrite('aftercut.jpg', cut_img)
        flag=1

def collectBGR():
    global points,flag
    img = cv2.imread("aftercut.jpg", 1)
    x_step = img.shape[1] / 8
    y_step = img.shape[0] / 12
    x_cr = x_step / 2
    y_cr = y_step / 2
    points = np.zeros((12, 8, 3), dtype=int)
    count = 0
    for i in range(1, 13):
        for j in range(1, 9):
            points[i - 1, j - 1] = (img[int(round(y_cr)), int(round(x_cr))])
            count += 1
            x_cr = x_cr + x_step
        y_cr = y_cr + y_step
        x_cr = x_step / 2
    flag = 2
    # print(points)

def calcandsave():
    global points
    dataSet, labels = kNN.loadDataSet()
    result = np.zeros((12,8))
    for i in range(12):
        for j in range(8):
            result[i,j] = kNN.classify0(points[i,j],np.array(dataSet),labels)
    df = pandas.DataFrame(result)
    df.to_csv('result.csv', header=False)
    print("Done!")

def cut():
    global img
    img = cv2.imread('test.jpg',1)
    cv2.namedWindow('image',0)
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    global flag
    flag = 0
    cut()
    if flag==1:
        collectBGR()
    if flag == 2:
        calcandsave()