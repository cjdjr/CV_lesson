import random
import shutil
import os
import matplotlib.image as mpimg
import cv2
if __name__=="__main__":

    # f_list = os.listdir("jpg/")
    # h=0
    # w=0
    # num=0
    # for i in f_list:
    #     if os.path.splitext(i)[1] == '.jpg':
    #         num+=1
    #         img=cv2.imread("jpg/"+i)
    #         h+=img.shape[0]
    #         w+=img.shape[1]
    # h=int(h/num)
    # w=int(w/num)
    # print(h,',',w)

    h=256
    w=256

    random.seed(19260817)
    for type in range(1,18):
        index=[i for i in range(1,81)]
        random.shuffle(index)
        base=(type-1)*80
        for i in range(60):
            name="image_"+str(base+index[i]).zfill(4)
            img=cv2.imread("jpg/"+name+".jpg")
            img=cv2.resize(img,(h,w))
            cv2.imwrite("train_set/"+name+".jpg",img)
            with open("train_set/"+name+".txt", 'w') as f:
                f.write(str(type))

        for i in range(60,80):
            name = "image_" + str(base + index[i]).zfill(4)
            img=cv2.imread("jpg/"+name+".jpg")
            img=cv2.resize(img,(h,w))
            cv2.imwrite("test_set/"+name+".jpg",img)
            with open("test_set/"+name+".txt", 'w') as f:
                f.write(str(type))
    