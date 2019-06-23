import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import random

patch_size=8
patch_num=100000
features_num=1200
random.seed(19260817)

def load_train_set(dir):
    f_list = os.listdir(dir)
    image=[]
    label=[]
    for i in f_list:
        print("load "+i)
        if os.path.splitext(i)[1] == '.jpg':
            image.append(mpimg.imread(dir+i))
        else:
            with open(dir+i, 'r') as f:
                l=f.readline()
            label.append(int(l)-1)

    image=np.array(image)
    label=np.array(label)
    label_one_hot=np.eye(17)[label.reshape(-1)]

    # random_shuffle dataset

    index=np.arange(len(image))
    random.shuffle(index)
    image=image[index]
    label_one_hot=label_one_hot[index]

    return image,label_one_hot

def sample_patch(image):
    X = np.zeros((patch_num, patch_size * patch_size * 3))
    for i in range(patch_num):
        print("sample_patch "+str(i))
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

def pre_process(X):
    mu=np.zeros((1,X.shape[1]))
    sigma=np.ones((1,X.shape[1]))
    P=np.eye(X.shape[1])

    # to do : whitening


    return X,mu,sigma,P

def K_means(X):
    centroid=np.zeros((features_num,X.shape[1]))

    # to do : K_means


    return centroid

def img2vec(img,mu,sigma,P):

    vector=np.zeros((features_num,1))
    # to do : convert image to feature vector

    return vector



def cal_image_feature(image,centroid,mu,sigma,P):

    image_feature=np.zeros((len(image),4*features_num))  # 4*features_num is an example
    # to do : calculate image feature by convolution (you may use img2vec)

    return image_feature

def build_classifier(image_feature,label):
    parameters=np.zeros((17,image_feature.shape[1]))
    # to do : build a softmax classifier


    return parameters

def save_parameters(file_name,mu,P,sigma,centroid,softmax_parameters):

    # to do : write mu to file

    # to do : write P and sigma to file

    # to do : write centroid to file

    # to do : write softmax_parameters to file

    return

if __name__=="__main__":


    image,label=load_train_set("train_set/")
    print("load finish!")

    X=sample_patch(image)
    print("sample patch finish!")

    X,mu,sigma,P=pre_process(X)

    assert(mu.shape[0]==1 and mu.shape[1]==X.shape[1])
    assert(sigma.shape[0]==1 and sigma.shape[1]==X.shape[1])

    centroid=K_means(X)
    assert(centroid.shape[0]==features_num and centroid.shape[1]==X.shape[1])

    image_feature=cal_image_feature(image,centroid,mu,sigma,P)
    assert(image_feature.shape[0]==len(image))

    print(image_feature.shape)

    parameters=build_classifier(image_feature,label)

    save_parameters("parameters.txt",mu,P,sigma,centroid,parameters)






