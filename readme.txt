
Geometric-Convolutional Feature Fusion Based on Learning Propagation for Facial Expression Recognition
June 28, 2018
Author: Yan Tang
All Right Reserved

Installation:
--------------------
1.Python 3.5 
2.Dlib 19.40
3.Tensorflow 1.2.1
4. other.....

Quick start: follow the step exactly
-------------------

1. To redo our work, please prepare the CK+ and Oulu-CASIA data accoding to the grouping listed 
in ./Data-Experiment/data_ck+/CK+_Ten_group.txt and
./Data-Experiment/data_OuluCasIA_VL_Strong/Oulu_CASIA_Ten_group.txt.
	For each line in txt file:"groupID    image Relative Path    label"
According to the data user license agreement, data is not allowed to share here. 
Our MMI dataset is sampling accroding to the CK+ and Oulu-CASIA
and regroup to 10 groups, due to the license, it's difficult to redo our work in MMI database. 
So the mmi productpkl.py file is not prepared here.
2. This step is prepared for landmarks detection. Download the dat file from 
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
and unzip the shape_predictor_68_face_landmarks.dat to 'dlibmodel' directory. 
3. Open the command line.
4. Use 'cd' command to reach the root of this directory.
5. Based on the txt file below, change the variable 'data_root_path'  to your data root 
in file 'productPKLforCKplus.py' and 'productPKLforOuluCasIA.py'
Type command 'python productPKLforCKplus.py' to pre-process ck+ data and save in a '.pkl' file.
Type command 'python productPKLforOuluCasIA.py' to pre-process oulu-casia data and save in a '.pkl' file.
The pre-processed face will be generated in ./Data-Experiment/face_from_ck+ or face_from_oulu directory
The reason for this step is we can pre-process only one time for convenience during our researches. 
For saving time, we can load data directly from the generated pkl file instead of pre-processing again.
6. Type command 'python cnn_for_fera_ten_fold_ten.py DatasetID NetworkID' to train and test our model
under ten-fold cross validation.
	Dataset ID: 1== MMI ; 2 == CK+ ; 3 == Oulu-CASIA
	Network ID: 1== DGFN ; 2 == DFSN ; 3 == DFSN-I
	As mentioned in 1 MMI is not prepared in this situation.
The result will be shown in the end of training and also shown in ./cnn_mark directory
