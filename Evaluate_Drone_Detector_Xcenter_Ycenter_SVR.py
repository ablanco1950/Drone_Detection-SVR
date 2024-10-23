# Author Alfonso Blanco

import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

import os
import re

import pickle #to save, load  the model

# valid is test
dirname="Drone-Detection-data-set(yolov7)-1\\valid\\images"
dirnameLabels="Drone-Detection-data-set(yolov7)-1\\valid\\labels"



########################################################################
def loadimages(dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     imgpathlabels = dirnameLabels + "\\"
     images = []
     TabFileName=[]
     TabImagesCV=[]
    
     print("Reading imagenes from ",imgpath)
     
     
     cont=0
     contRejected=0
     Yxmidpoint=[]
     Yymidpoint=[]
     Ywmidpoint=[]
     Yhmidpoint=[]
     TabTrueBoxesResized=[]
     
     for root, dirnames, filenames in os.walk(imgpath):
        
         
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                                 
                 image = cv2.imread(filepath)

                 filenameLabel=filename[0:len(filename)-3]+ "txt"
                 filenameLabel=imgpathlabels + filenameLabel
                 #print( filenameLabel)
                                
                 f=open(filenameLabel,"r")
                                
                 xywh=""
                 SwEmpty=0
                 for linea in f:
                 
                      cont= cont +1
                     
                      xywh=linea[2:]
                      
                      
                      SwEmpty=1
                      xywh=xywh.split(" ")
                      #plot_image(image, [], xywh)
                      Yxmidpoint.append(str(float(xywh[0])))
                      Yymidpoint.append(str(float(xywh[1])))
                      Ywmidpoint.append(str(float(xywh[2])))
                      Yhmidpoint.append(str(float(xywh[3])))
                      break # only one drone per image is considered 
                 
                 TabImagesCV.append(image)
                 # TO REDUCE MEMORY PROBLEMS, CONVERT TO GRAY
                 cv2.imwrite("pptest.jpg", image)

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


    # Create a Rectangle patch
    Cont=0
    #print(box)
    
    upper_left_x_True = float(boxTrue[0]) - float( boxTrue[2] )/ 2.0
    upper_left_y_True = float(boxTrue[1]) - float( boxTrue[3]) / 2.0
    rect = patches.Rectangle(
            (upper_left_x_True * width, upper_left_y_True * height),
            float(boxTrue[2]) * width,
            float(boxTrue[3]) * height,
            linewidth=2,
            edgecolor="green",
            facecolor="none",
        )
        # Add the patch to the Axes
       
    ax.add_patch(rect)
   
    plt.show()

###########################################################
# MAIN
##########################################################


imagesCV, X_test, TabFileName, Yxmidpoint, Yymidpoint, Ywmidpoint, Yhmidpoint=loadimages(dirname)

print("Number of images to test : " + str(len(X_test)))


y_test_pred_Yxmidpoint=[]
f=open("Predicted_True_Xcenter_Drone_Detector.txt","r")

for linea in f:                   
                        
  
  xcentered=linea.split(",")
  y_test_pred_Yxmidpoint.append(float(xcentered[0]))
  
y_test_pred_Yymidpoint=[]
f=open("Predicted_True_Ycenter_Drone_Detector.txt","r")

for linea in f:                   
  
  ycentered=linea.split(",")
  y_test_pred_Yymidpoint.append(float(ycentered[0]))

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

print("===")

print("==============================================================================")
for i in range (len(imagesCV)):
    img=imagesCV[i]
   
    height, width, _ = img.shape
   
    p1=float(y_test_pred_Yxmidpoint[i])* float(width)
    p1=int(p1)
    
    p2=float(y_test_pred_Yymidpoint[i])* float(height)
    p2=int(p2)
   
    cv2.circle(img,(p1,p2), 40, (0,0,255), thickness=5)
        
    boxes=[]
    boxesTrue=[]
    boxesTrue.append(Yxmidpoint[i])
    boxesTrue.append(Yymidpoint[i])
    boxesTrue.append(Ywmidpoint[i])
    boxesTrue.append(Yhmidpoint[i])
   
    plot_image(img, boxes, boxesTrue, TabFileName[i])
    

