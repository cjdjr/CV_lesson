import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import random

patch_size=8
patch_num=100000
image=[]
label=[]
X=[]

def load_train_set():
    f_list = os.listdir("train_set/")
    image=[]
    label=[]
    for i in f_list:
        print("load "+i)
        if os.path.splitext(i)[1] == '.jpg':
            image.append(mpimg.imread("train_set/"+i))
        else:
            with open("train_set/"+i, 'r') as f:
                l=f.readline()
            label.append(int(l))
    return image,label

def cal_X():
    X = np.zeros((patch_num, patch_size * patch_size * 3))
    for i in range(patch_num):
        print("cal_X"+str(i))
        index=random.randint(0,len(image)-1)
        h=image[index].shape[0]
        w=image[index].shape[1]
        base_x=random.randint(0,h-patch_size)
        base_y=random.randint(0,w-patch_size)
        now=0
        for x in range(patch_size):
            for y in range(patch_size):
                X[i][now]=image[index][base_x+x][base_y+y][0]
                now+=1
                X[i][now] = image[index][base_x + x][base_y + y][1]
                now+=1
                X[i][now] = image[index][base_x + x][base_y + y][2]
                now+=1
    return X


if __name__=="__main__":

    random.seed(19260817)
    image,label=load_train_set()
    print("load finish!")

    cal_X()
    print("cal_X finish!")







