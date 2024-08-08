# Drone_Detection-SVR
From dataset https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1# a model is obtained, based on ML (SVR), with that custom dataset, to indicate drones detection

The train file should be downloaded from https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1#
(needs a roboflow key)

Requirements:

Download all project datasets to a folder on disk.

You only need to have 4GB of free disk space (which is usual), you may need more than 16GB of RAM

You don't need a GPU

But because of this RAM limitation, you have to run the process in several steps:

1-
Create a model that obtains the Xcenter coordinate of each drone, by running:

Train_Drone_Detector_Yxmidpoint-SVR.py

it takes less than 30 minutess and creates the svr_lin_Drone_Detector_Yxmidpoint.pickle  model with a size of about 2 Gb

2-
Create a model that obtains the Ycenter coordinate of each drone, by running:

Train_Drone_Detector_Yymidpoint-SVR.py

it takes less than 30 minutesrs and creates the svr_lin_Drone_Detector_Yymidpoint.pickle model with a size of about 2 Gb

3-
From the test images ( is used the valid folder as test)  and labels  and based on the svr_lin_Drone_Detector_Yxmidpoint.pickle  model obtained in step 1- , create a .txt file with the predicted Xcenter coordinates for each dron of the test images (Predicted_True_Xcenter_Drone_Detector.txt file)

by running:

Create_File_Drone_Detector_With_Predicted_Xcenter_SVR.py

4-

From the test images ( is used the valid folder as test)  and labels  and based on the svr_lin_Drone_Detector_Yymidpoint.pickle  model obtained in step 2- , create a .txt file with the predicted Ycenter coordinates for each dron of the test images (Predicted_True_Ycenter_Drone_Detector.txt file)

by running:

Create_File_Drone_Detector_With_Predicted_Ycenter_SVR.py


5- Execute an  Evaluation by running:

Evaluate_Resized_bone-fracture-2_Xcenter_Ycenter_SVR.py

The test images appear on the screen with a blue circle indicating the predicted dron and a green rectangle indicating where the dron  was indicated when the image was labeled.


References:

https://universe.roboflow.com/drone-detection-pexej/drone-detection-data-set-yolov7/dataset/1#

https://medium.com/@niousha.rf/support-vector-regressor-theory-and-coding-exercise-in-python-ca6a7dfda927

https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
