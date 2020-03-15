import cv2,os
import imutils
import numpy as np

cv2.namedWindow('panel2')
cv2.namedWindow('panel1')
cv2.namedWindow('panel')

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

all_points=[]

def fun(x):
    pass
cv2.createTrackbar('l_h','panel1',13,255,fun)
cv2.createTrackbar('l_s','panel1',167,255,fun)
cv2.createTrackbar('l_v','panel1',204,255,fun)
cv2.createTrackbar('h_h','panel1',176,255,fun)
cv2.createTrackbar('h_s','panel1',255,255,fun)
cv2.createTrackbar('h_v','panel1',255,255,fun)


kernel=np.ones((5,5))
final_points=[]
clusters=[]
dont=[]
Y=[]
#kernelsharp = np.array([[-1,-1,-1], [-1,15,-1], [-1,-1,-1]])
for name in os.listdir('retino\\'):

    img=cv2.imread('retino\\{}'.format(name))
    l_h=cv2.getTrackbarPos('l_h','panel1')
    l_s=cv2.getTrackbarPos('l_s','panel1')
    l_v=cv2.getTrackbarPos('l_v','panel1')
    h_h=cv2.getTrackbarPos('h_h','panel1')
    h_s=cv2.getTrackbarPos('h_s','panel1')
    h_v=cv2.getTrackbarPos('h_v','panel1')

    lower,higher=(l_h,l_s,l_v),(h_h,h_s,h_v)

    img=cv2.resize(img,(640,480))
    org=img.copy()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.bitwise_not(gray)
    new_img=adjust_gamma(gray,0.348)
    new_img=cv2.bitwise_not(new_img)
    new_img=adjust_gamma(new_img,0.125)
    new_img=cv2.bitwise_not(new_img)
    new_img=adjust_gamma(new_img,0.07)

    cnts=cv2.findContours(new_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)

    if len(cnts)>0:
        for _ in range(3):
            contour=max(cnts, key = lambda m :cv2.contourArea(m))
            (x,y,w,h)=cv2.boundingRect(contour)
            dont.append((x,y,w,h))

        for c in cnts:
            (x,y,w,h)=cv2.boundingRect(c)
            if w*h>20 and (w>3 and h>3) and (w<65 and h<65) and (x,y,w,h) not in dont:

                cv2.rectangle(org,(x,y),(x+w,y+h),(255,255,255),-1)
                #finding cutters
                if len(clusters)==0:
                    clusters.append((x,y,w,h))
                else:
                    for cluster in clusters:
                        px,py,pw,ph=cluster
                        if (x>px and x<px+pw) and (y>py and y<py+ph) and (x,y,w,h) not in clusters:
                            clusters.append((x,y,w,h))
                            break


    final_image=org.copy()
    final_image_gray=cv2.cvtColor(final_image,cv2.COLOR_BGR2GRAY)
    final_image_gray=cv2.GaussianBlur(final_image_gray,(17,17),0)
    _,final_image_thresh=cv2.threshold(final_image_gray,225,255,cv2.THRESH_BINARY_INV)
    final_cnts=cv2.findContours(final_image_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    final_cnts=imutils.grab_contours(final_cnts)
    count=0
    for c in final_cnts:
        (x,y,w,h)=cv2.boundingRect(c)
        points=((x+x+w)/2,(y+y+h)/2)
        all_points.append(points)
        if (w>=15 and h>=15):

            count+=1
    final_points.append(all_points)
    all_points=[]
    print(name,count)
    count=len(final_cnts)
    Y.append(count)
    """    if count>=2:
        Y.append(1)
    else:
        Y.append(0)
    """
    cv2.imshow('final_img',new_img)
    cv2.imshow('img',img)
    cv2.imshow('final_img_thresh',final_image_thresh)

    if cv2.waitKey(1)==ord('q'):
        break

import pickle
f=open('points.pkl','wb')
pickle.dump(final_points,f)
f.close()
