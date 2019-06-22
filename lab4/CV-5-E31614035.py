import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def region_of_interest(img, vertices):

    '''
    :param img: 原始图片
    :param vertices: 感兴趣的区域
    :return: bitwise_and后的结果
    '''
    print(vertices)
    mask = np.zeros_like(img)
    match_mask_color = 255
    #print(match_mask_color)

    cv2.fillPoly(mask, vertices, match_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hls_select(img,channel='s',thresh=(90, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    plt.figure()
    plt.imshow(hls)
    if channel=='h':
        channel = hls[:,:,0]
    elif channel=='l':
        channel=hls[:,:,1]
    else:
        channel=hls[:,:,2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):

    if lines is None:
        return None
    print("ok")
    img = np.copy(img)
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    print("ok")
    return img

def pick_correct_line(lines):
    ans=[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1==x2 or abs((y2-y1)/(x2-x1))>0.5:
                ans.append(line)

    return ans

if __name__=="__main__":
    # reading in an image
    image = mpimg.imread('./车道图片/00226_00001.jpg')
    # printing out some stats and plotting the image
    print('This image is:', type(image), 'with dimensions:', image.shape)
    plt.imshow(image)
    #plt.show()
    Gauss_img = cv2.GaussianBlur(image, (3,3), 1.5)


    l_binary = hls_select(Gauss_img, channel='l', thresh=(80, 230))
    #s_binary = hls_select(Gauss_img, channel='s', thresh=(100, 255))

#    select_lane_pix=l_binary&s_binary
    #plt.figure()
    #plt.imshow(select_lane_pix)
    #print(select_lane_pix)
    gray_image = cv2.cvtColor(Gauss_img, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 30, 80)
    cannyed_image[(l_binary == 0)] = 0
    plt.figure("canny_image")
    plt.imshow(cannyed_image)

    region_of_interest_vertices = [
        (image.shape[1]/6, image.shape[0]/3),
        (image.shape[1]*5/6, image.shape[0]/3),
        (image.shape[1],image.shape[0]),
        (0,image.shape[0]),
    ]
    cropped_image = region_of_interest(cannyed_image,np.array([region_of_interest_vertices], np.int32))
    plt.figure("cropped_image")
    plt.imshow(cropped_image)


    #gray_image[(l_binary == 0)] = 0
    #plt.figure("gray_image")
    #plt.imshow(gray_image)


    lines = cv2.HoughLinesP(
        cropped_image,
        rho=1,              # delta rho
        theta=np.pi / 60,   # delta theta
        threshold=40,       # 投票阈值数
        lines=np.array([]),
        minLineLength=400,   # 最小的线段长度，小于该长度的线段将会被丢弃
        maxLineGap=200      # 一条线段上的两个点的最大距离
    )
    lines=pick_correct_line(lines)
    print(lines)
    line_image = draw_lines(image, lines)  # <---- Add this call.
    plt.figure("line_image")
    plt.imshow(line_image)

    #print(lines)

    plt.show()