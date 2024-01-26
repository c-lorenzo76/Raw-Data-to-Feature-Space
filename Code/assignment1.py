# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:40:46 2023

@author: Cristian
"""

import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# read files 
cardinal = cv2.imread("C:/Users/Cristin/Desktop/410_assignment_2/bird images/image0.jpg")
cardinalL = cv2.imread("C:/Users/Cristin/Desktop/410_assignment_2/bird images/Cardinal_0006_17684.png")

sparrow = cv2.imread("C:/Users/Cristin/Desktop/410_assignment_2/bird images/image1.jpg")
sparrowL = cv2.imread("C:/Users/Cristin/Desktop/410_assignment_2/bird images/Field_Sparrow_0013_113599.png")
                      
crow = cv2.imread("C:/Users/Cristin/Desktop/410_assignment_2/bird images/image2.jpg")
crowL = cv2.imread("C:/Users/Cristin/Desktop/410_assignment_2/bird images/American_Crow_0016_25112.png")

# color channels
plt.imshow(cardinal[:,:,0])
plt.imshow(cardinal[:,:,1])
plt.imshow(cardinal[:,:,2])

plt.imshow(cardinalL[:,:,0])
plt.imshow(cardinalL[:,:,1])
plt.imshow(cardinalL[:,:,2])

plt.imshow(sparrow[:,:,0])
plt.imshow(sparrow[:,:,1])
plt.imshow(sparrow[:,:,2])

plt.imshow(sparrowL[:,:,0])
plt.imshow(sparrowL[:,:,1])
plt.imshow(sparrowL[:,:,2])

plt.imshow(crow[:,:,0])
plt.imshow(crow[:,:,1])
plt.imshow(crow[:,:,2])

plt.imshow(crowL[:,:,0])
plt.imshow(crowL[:,:,1])
plt.imshow(crowL[:,:,2])


# convert to grayscale - cardinal

cardinalG = cv2.cvtColor(cardinal, cv2.COLOR_BGR2GRAY)
heightcG, widthcG = cardinalG.shape

cardinalLG = cv2.cvtColor(cardinalL, cv2.COLOR_BGR2GRAY)
heightclG, widthclG = cardinalLG.shape

# convert to grayscale - sparrow
sparrowG = cv2.cvtColor(sparrow, cv2.COLOR_BGR2GRAY)
heightsG, widthsG = sparrowG.shape

sparrowLG = cv2.cvtColor(sparrowL, cv2.COLOR_BGR2GRAY)
heightslG, widthslG = sparrowLG.shape

# convert to grayscale - crow
crowG = cv2.cvtColor(crow, cv2.COLOR_BGR2GRAY)
heightcrG, widthccrG = crowG.shape

crowLG = cv2.cvtColor(crowL, cv2.COLOR_BGR2GRAY)
heightcrlG, widthcrlG = crowLG.shape

# display cardinal grayscale

plt.imshow(cardinalG, cmap=plt.get_cmap('gray'))
plt.axis('off')
print('dimension: ', cardinalG.shape)

plt.imshow(cardinalLG, cmap=plt.get_cmap('gray'))
plt.axis('off')
print('dimension: ', cardinalLG.shape)

plt.imshow(sparrowG, cmap=plt.get_cmap('gray'))
plt.axis('off')
print('dimension: ', sparrowG.shape)

plt.imshow(sparrowLG, cmap=plt.get_cmap('gray'))
plt.axis('off')
print('dimension: ', sparrowLG.shape)

plt.imshow(crowG, cmap=plt.get_cmap('gray'))
plt.axis('off')
print('dimension: ', crowG.shape)

plt.imshow(crowLG, cmap=plt.get_cmap('gray'))
plt.axis('off')
print('dimension: ', crowLG.shape)


# Raw data (image) resizing

# resizing - Cardinal
cardC = cv2.resize(cardinalG, dsize=(320,240), interpolation=cv2.INTER_CUBIC)
cardCL = cv2.resize(cardinalLG, dsize=(320,240), interpolation=cv2.INTER_CUBIC)

card1 = cv2.normalize(cardC.astype('float'), None, 0.0, 1.0,
cv2.NORM_MINMAX)*255

cardL1 = cv2.normalize(cardCL.astype('float'), None, 0.0, 1.0,
cv2.NORM_MINMAX)*255

heightc, widthc = card1.shape
heightb, widthb = cardL1.shape

# resizing - Sparrow
cardS = cv2.resize(sparrowG, dsize=(320,240), interpolation=cv2.INTER_CUBIC)
cardSL = cv2.resize(sparrowLG, dsize=(320,240), interpolation=cv2.INTER_CUBIC)

card2 = cv2.normalize(cardS.astype('float'), None, 0.0, 1.0,
cv2.NORM_MINMAX)*255

cardL2 = cv2.normalize(cardSL.astype('float'), None, 0.0, 1.0,
cv2.NORM_MINMAX)*255

heightd, widthd = card2.shape
heighte, widthe = cardL2.shape

# resizing - Crow
cardCR = cv2.resize(crowG, dsize=(320,240), interpolation=cv2.INTER_CUBIC)
cardCRL = cv2.resize(crowLG, dsize=(320,240), interpolation=cv2.INTER_CUBIC)

card3 = cv2.normalize(cardCR.astype('float'), None, 0.0, 1.0,
cv2.NORM_MINMAX)*255

cardL3 = cv2.normalize(cardCRL.astype('float'), None, 0.0, 1.0,
cv2.NORM_MINMAX)*255

heightf, widthf = card3.shape
heightg, widthg = cardL3.shape



# plot the images
plt.imshow(card1, cmap=plt.get_cmap('gray'))
plt.axis('off')

plt.imshow(cardL1, cmap=plt.get_cmap('gray'))
plt.axis('off')

plt.imshow(card2, cmap=plt.get_cmap('gray'))
plt.axis('off')

plt.imshow(cardL2, cmap=plt.get_cmap('gray'))
plt.axis('off')

plt.imshow(card3, cmap=plt.get_cmap('gray'))
plt.axis('off')

plt.imshow(cardL3, cmap=plt.get_cmap('gray'))
plt.axis('off')

# Save images to png
cv2.imwrite('C:/Users/Cristin/Desktop/410_assignment_3/cardC_grayTD.png',cardC)
cv2.imwrite('C:/Users/Cristin/Desktop/410_assignment_3/cardCL_grayTD.png',cardCL)
cv2.imwrite('C:/Users/Cristin/Desktop/410_assignment_3/cardC_seg.png',cardC*(cardCL/255))

cv2.imwrite('C:/Users/Cristin/Desktop/410_assignment_3/cardS_grayTD.png',cardS)
cv2.imwrite('C:/Users/Cristin/Desktop/410_assignment_3/cardSL_grayTD.png',cardSL)
cv2.imwrite('C:/Users/Cristin/Desktop/410_assignment_3/cardS_seg.png',cardS*(cardSL/255))

cv2.imwrite('C:/Users/Cristin/Desktop/410_assignment_3/cardCR_grayTD.png',cardCR)
cv2.imwrite('C:/Users/Cristin/Desktop/410_assignment_3/cardCRL_grayTD.png',cardCRL)
cv2.imwrite('C:/Users/Cristin/Desktop/410_assignment_3/cardCR_seg.png',cardCR*(cardCRL/255))


##############################################################################
# Overlapping Cardinal: 
card_seg = cardC *(cardCL/255)
cc = round(((heightc)*(widthc)))
flatcc = np.zeros((cc, 256), np.uint16)
k = 0
ss = []
for i in range(heightc-15):
    for j in range(widthc-15):
        crop_tmp = card_seg[i:i+16,j:j+16]
        if (crop_tmp.sum() !=0):
            flatcc[k,0:256] = crop_tmp.flatten()
            flatcc[k,255] = 1 
            tmp = flatcc[k,0:257]
            ss.append(tmp)
            k = k + 1

fspaceCC = pd.DataFrame(ss) #panda object
fspaceCC.to_csv('C:/Users/Cristin/Desktop/410_assignment_3/fspaces/fspaceI0.csv', index=False)


##############################################################################
# Overlapping Sparrow 
card_seg = cardS *(cardSL/255)
cc = round(((heightd)*(widthd)))
flatcc = np.zeros((cc, 256), np.uint16)
k = 0
ss = []
for i in range(heightd-15):
    for j in range(widthd-15):
        crop_tmp = card_seg[i:i+16,j:j+16]
        if (crop_tmp.sum() !=0):
            flatcc[k,0:256] = crop_tmp.flatten()
            flatcc[k,255] = 2
            tmp = flatcc[k,0:257]
            ss.append(tmp)
            k = k + 1

fspaceSS = pd.DataFrame(ss) #panda object
fspaceSS.to_csv('C:/Users/Cristin/Desktop/410_assignment_3/fspaces/fspaceI1.csv', index=False)

##############################################################################
# Overlapping Crow 
card_seg = cardCR *(cardCRL/255)
cc = round(((heightf)*(widthf)))
flatcc = np.zeros((cc, 256), np.uint16)
k = 0
ss = []
for i in range(heightf-15):
    for j in range(widthf-15):
        crop_tmp = card_seg[i:i+16,j:j+16]
        if (crop_tmp.sum() !=0):
            flatcc[k,0:256] = crop_tmp.flatten()
            flatcc[k,255] = 3
            tmp = flatcc[k,0:257]
            ss.append(tmp)
            k = k + 1

fspaceCR = pd.DataFrame(ss) #panda object
fspaceCR.to_csv('C:/Users/Cristin/Desktop/410_assignment_3/fspaces/fspaceI2.csv', index=False)

#################################################################
# Join the feature vectors of the classes Cardinal and Sparrow

frames = [fspaceCC,fspaceSS]
mged = pd.concat(frames)
indx = np.arange(len(mged))
rndmged = np.random.permutation(indx)
rndmged=mged.sample(frac=1).reset_index(drop=True)
rndmged.to_csv('C:/Users/Cristin/Desktop/410_assignment_3/fspaces/merged01.csv',index=False)

#################################################################
# Join the feature vectors of the classes Crow and Sparrow

frames = [fspaceCR,fspaceSS]
mged = pd.concat(frames)
indx = np.arange(len(mged))
rndmged = np.random.permutation(indx)
rndmged=mged.sample(frac=1).reset_index(drop=True)
rndmged.to_csv('C:/Users/Cristin/Desktop/410_assignment_3/fspaces/merged12.csv',index=False)

#################################################################
# Join the feature vectors of the classes Cardinal and Crow

frames = [fspaceCC,fspaceCR]
mged = pd.concat(frames)
indx = np.arange(len(mged))
rndmged = np.random.permutation(indx)
rndmged=mged.sample(frac=1).reset_index(drop=True)
rndmged.to_csv('C:/Users/Cristin/Desktop/410_assignment_3/fspaces/merged02.csv',index=False)





































