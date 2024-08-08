# Author Alfonso Blanco

import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

import os
import re

import pickle #to save, load  the model


dirname="Test1"

imageSize=64

########################################################################
def loadimages(dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     TabFileName=[]
     TabImagesCV=[]
    
     print("Reading imagenes from ",imgpath)
     
     
     cont=0
     #contRejected=0
     Yxmidpoint=[]
     Yymidpoint=[]
     Ywmidpoint=[]
     Yhmidpoint=[]
     TabTrueBoxesResized=[]
     
     for root, dirnames, filenames in os.walk(imgpath):
        
         
         
        for filename in filenames:
             #print(filename)
             
             if re.search("\.(jpg|jpeg|JPEG|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                 #print(filepath)
                                 
                 image = cv2.imread(filepath)
                
                
                 result= cv2.resize(image, (imageSize,imageSize), interpolation = cv2.INTER_AREA)
                 
                 #image = cv2.resize(image, (imageSize,imageSize), interpolation = cv2.INTER_AREA)
                 TabImagesCV.append(image)
                 # TO REDUCE MEMORY PROBLEMS, CONVERT TO GRAY
                 cv2.imwrite("pptest.jpg", result)

                 result= cv2.imread("pptest.jpg", cv2.IMREAD_GRAYSCALE)
                 
                 result=result.flatten()
                 
                 images.append(result)
                 TabFileName.append(filename)
              
                 cont+=1
     
     return  TabImagesCV, images, TabFileName, Yxmidpoint, Yymidpoint, Ywmidpoint, Yhmidpoint

def plot_image(image, box, boxTrue, NameImage):
    
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    #Figure.suptitle(NameImage)
    fig.suptitle(NameImage)
    # Display the image
    ax.imshow(im)
  
       
    plt.show()

###########################################################
# MAIN
##########################################################


imagesCV, X_test, TabFileName, Yxmidpoint, Yymidpoint, Ywmidpoint, Yhmidpoint=loadimages(dirname)

print("Number of images to test : " + str(len(X_test)))

# https://medium.com/@niousha.rf/support-vector-regressor-theory-and-coding-exercise-in-python-ca6a7dfda927
from sklearn.preprocessing import StandardScaler

### When using StandardScaler(), fit() method expects a 2D array-like input
scaler = StandardScaler().fit(X_test)
X_test_scaled = scaler.transform(X_test)

model_svr_lin_Yxmidpoint= pickle.load( open("svr_lin_Drone_Detector_Yxmidpoint.pickle", 'rb'))
model_svr_lin_Yymidpoint= pickle.load( open("svr_lin_Drone_Detector_Yymidpoint.pickle", 'rb'))

import numpy as np
from sklearn import metrics

#### Test dataset - metrics ####
y_test_pred_Yxmidpoint = model_svr_lin_Yxmidpoint.predict(X_test_scaled)
y_test_pred_Yymidpoint = model_svr_lin_Yymidpoint.predict(X_test_scaled)



# https://medium.com/@niousha.rf/support-vector-regressor-theory-and-coding-exercise-in-python-ca6a7dfda927


print("Total Xcenter coordinates = " + str(len(y_test_pred_Yxmidpoint)))
print("TotalYcenter coordinates = " + str(len(y_test_pred_Yymidpoint)) )     
                      
print("predicted values for Xcenter:")
print(y_test_pred_Yxmidpoint)
print("true values for Xcenter:")
print(Yxmidpoint)
print("===")

print("predicted values for Ycenter:")
print(y_test_pred_Yymidpoint)
print("true values for Ycenter:")
print(Yymidpoint)


print("==============================================================================")
for i in range (len(imagesCV)):
    img=imagesCV[i]
    print(TabFileName[i])
    height, width, _ = img.shape
    #print(y_test_pred_Yxmidpoint[i])
    #print(width)
    p1=float(y_test_pred_Yxmidpoint[i])* float(width)
    p1=int(p1)
    #print(p1)
    #print(int(p1))
    p2=float(y_test_pred_Yymidpoint[i])* float(height)
    p2=int(p2)
    #print(p2)
    #print(int(p2))
    #print(y_test_pred_Yymidpoint[i])
    #print(height)
    #cv2.circle(img,int(p1),int(p2), 10, (0,255,0), thickness=5)
    cv2.circle(img,(p1,p2), 80, (0,0,255), thickness=5)
    #cv2.imshow("ROI", img)  
    #cv2.waitKey(0)
    
    boxes=[]
    boxesTrue=[]
    
    plot_image(img, boxes, boxesTrue, TabFileName[i])
    

