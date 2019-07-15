import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.utils import shuffle

def maxx(maxvec, fts, num):
    for i in range(0,num):
        if(abs(fts[i]) > maxvec[i]):
            maxvec[i] = abs(fts[i])
    return maxvec

def extractFeatures(obj, img):
    X, Y = 1, 0
    A, gx, gy = 0, 0, 0
    N = len(obj)

    # centroide e area
    for i in range(0, N-1):
        A += (obj[i][0][X] * obj[i+1][0][Y]) - (obj[i+1][0][X] * obj[i][0][Y])
        gx += (obj[i][0][X] + obj[i+1][0][X]) * (obj[i][0][X] * obj[i+1][0][Y] - obj[i+1][0][X] * obj[i][0][Y])
        gy += (obj[i][0][Y] + obj[i+1][0][Y]) * (obj[i][0][X] * obj[i+1][0][Y] - obj[i+1][0][X] * obj[i][0][Y])
        #M = cv2.moments(obj)
        #gx, gy = float(M['m10']/M['m00']), float(M['m01']/M['m00'])

    #gx = 28 + gx
    #gy = 28 + gy
    #A = cv2.contourArea(obj)

    if A != 0:
        gx *= 1/(3.0*A)
        gy *= 1/(3.0*A)
    else:
        gx = 0
        gy = 0
        print("Error! A == 0")


    # excentricidade e eixos x e y
    cxx, cxy, cyy = 0.0, 0.0, 0.0
    for i in range(0, N-1):
        cxx += (obj[i][0][X] - gx) ** 2
        cyy += (obj[i][0][Y] - gy) ** 2
        cxy += (obj[i][0][X] - gx) * (obj[i][0][Y] - gy)

    lmbda1 = (cxx + cyy + ( (cxx + cyy)**2 - 4.0*(cxx*cxy - cxy**2) )**(1/2) )
    lmbda2 = (cxx + cyy - ( (cxx + cyy)**2 - 4.0*(cxx*cxy - cxy**2) )**(1/2) )

    if(lmbda1 == 0):
        print("Error! lbda1 == 0")
        E = 0.0
    else:
        E = lmbda2/lmbda1


    # ajustar circulo
    (x,y),radius = cv2.minEnclosingCircle(obj)
    center = (int(x),int(y))
    radius = int(radius)

	# Convex hull
    hull = cv2.convexHull(obj, returnPoints = False)
    try:
        defects = cv2.convexityDefects(obj,hull)
        numdefects = len(defects)
    except:
        numdefects = 0

    #print(len(defects))

	# ajustar ellipse
    (x,y),(MA,ma),angle = cv2.fitEllipse(obj)

	# pontos de extremo
    leftmost = tuple(obj[obj[:,:,0].argmin()][0])
    rightmost = tuple(obj[obj[:,:,0].argmax()][0])
    topmost = tuple(obj[obj[:,:,1].argmin()][0])
    bottommost = tuple(obj[obj[:,:,1].argmax()][0])


	# teste de convexidade
    k = cv2.isContourConvex(obj)

    epsilon = 0.1*cv2.arcLength(obj,True)
    approx = cv2.approxPolyDP(obj,epsilon,True)
    #print(len(approx))

	# numero de linhas
    #edges = cv2.Canny(img,1, 1,apertureSize = 3)
    lines = cv2.HoughLines(img,1,np.pi/180,28)
    try:
        number_lines = len(lines)
    except:
        number_lines = 0

    # HU moments
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments)

	# append de features
    fts = []
    fts.append(abs(gx)) # menor threshold, maior o impacto
    fts.append(abs(gy))
    fts.append(A)
    fts.append(E)
    fts.append(lmbda1) # horizontal axis
    fts.append(lmbda2) # vertical axis
    fts.append(radius)
    fts.append(center[0])
    fts.append(center[1])
    fts.append(center[0] - gx)
    fts.append(center[1] - gy) # 10
    fts.append(len(hull))
    fts.append(angle)
    fts.append(rightmost[0] - center[0])
    fts.append(rightmost[1] - center[1])
    fts.append(leftmost[0] - center[0])
    fts.append(leftmost[1] - center[1])
    fts.append(k) # 17
    fts.append(numdefects)
    fts.append(number_lines)
    for i in hu:
        fts.append(i[0])


    return fts

def loadFeatures(X, Y, DATAPATH):
    THRESH = 1
    maxvec = []
    for img_name in tqdm(os.listdir(DATAPATH)):

        img = cv2.imread(os.path.join(DATAPATH, img_name), cv2.IMREAD_GRAYSCALE)

        img_bw = cv2.threshold(img, THRESH, 255, 0)[1]
        contours, hierarchy  = cv2.findContours(img_bw, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]

        if len(contours) >= 1:
            for i in range(0, len(contours)):
                if(len(contours[i]) > len(cnt)):
                    cnt = contours[i]
            #a = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
            #cv2.imshow('draw contours', a)
            #cv2.waitKey(0)

        #a = cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
        #cv2.imshow("aa", a)
        #cv2.waitKey(0

        fts = extractFeatures(cnt, img_bw)
        if len(maxvec) == 0:
            maxvec= [-1 for i in range(0, len(fts))]
        maxvec = maxx(maxvec, fts, len(fts))
        X.append(fts)
        Y.append(int(img_name[4]))

    # divide os dados pelo maior valor absoluto para manter todos entre -1 e 1
    for i in range(0, len(X)):
        for j in range(0, len(X[i])):
            if(maxvec[j] != 0):
                X[i][j] = X[i][j]/float(maxvec[j])
