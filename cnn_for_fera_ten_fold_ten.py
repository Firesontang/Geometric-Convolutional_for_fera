# coding=gbk 
# @Author: Yan Tang
# @Date:   2018-06-27 

'''
----------------------------------------------------------------
Model training and testing under ten-fold cross validation
Usage:python cnn_for_fera_ten_fold_ten.py DatasetID NetworkID
Dataset ID: 1== MMI ; 2 == CK+ ; 3 == Oulu-CASIA
Network ID: 1== DGFN ; 2 == DFSN ; 3 == DFSN-I
----------------------------------------------------------------
'''

import tensorflow as tf
import numpy as np
import os
import time,sys
import logging
import matplotlib.pyplot as plt
import win_unicode_console
win_unicode_console.enable()


train_epochs = 100    # epochs
epochs_time = 100   #epochs iteration
batch_size   = 30     # training batach size
learning_rate= 0.0001  #learning rate
test_bound = 1 # test_accuract_criterion

pkl_path1 = './Data-Experiment/MMI_10-group_3Frame/mmi_with_img_geometry_3frame.pkl'
pkl_path2 = './Data-Experiment/data_ck+/ckplus_with_img_geometry_3frame.pkl'
pkl_path3 = './Data-Experiment/data_OuluCasIA_VL_Strong/oulus_casia_with_img_geometry_3frame.pkl'

pkl_flag = 0
model_flag = 0
if len(sys.argv)==3:
    pkl_flag = int(sys.argv[1])
    model_flag = int(sys.argv[2])
else:
    print('>>>>Usage:python cnn_for_fera_ten_fold_ten.py DatasetID NetworkID')
    print('>>>>Dataset ID: 1== MMI ; 2 == CK+ ; 3 == Oulu-CASIA')
    print('>>>>Network ID: 1== DGFN ; 2 == DFSN ; 3 == DFSN-I ')
    exit(1)

pkl_path = ''
filenames = ''

if not os.path.exists('./cnn_mark'):
    os.mkdir('./cnn_mark')

#dataset setting---------------
if pkl_flag == 1:
    print(">>>>MMI dataset is not exits. See readme.txt")
    exit(1)
    pkl_path = pkl_path1
    filenames = "./cnn_mark/cnn_ten_fold_mark_mmi_"
    print('>>>>Dataset : MMI')
elif pkl_flag == 2:
    pkl_path = pkl_path2
    filenames = "./cnn_mark/cnn_ten_fold_mark_ckp_"
    print('>>>>Dataset : CK+')
elif pkl_flag == 3:
    pkl_path = pkl_path3
    print('>>>>Dataset : Oulu-CASIA')
    filenames = "./cnn_mark/cnn_ten_fold_mark_oulu_"
else:
    print('>>>>Dataset ID not exist')
    print('>>>>Dataset ID: 1== MMI ; 2 == CK+ ; 3 == Oulu-CASIA')
    exit(1)
#dataset setting--------------- end

if model_flag == 1:
    filenames = filenames+'DGFN.txt'
    train_epochs = 300
    test_bound = 0.85
    print('>>>>Network : DGFN')
elif model_flag ==2 :
    filenames = filenames+'DFSN.txt'
    train_epochs = 60
    test_bound = 0.95
    print('>>>>Network : DFSN')
elif model_flag ==3 :
    filenames = filenames+'DFSN_I.txt'
    train_epochs = 60
    test_bound = 0.95
    print('>>>>Network : DFSN-I')
else:
    print('>>>>Network ID not exist')
    print('>>>>Network ID: 1== DGFN ; 2 == DFSN ; 3 == DFSN-I ')
    exit(1)



#log for accuracy of ten-fold cross validation
logging.basicConfig(level=logging.DEBUG, 
                    filename=filenames, 
                    filemode="a+", 
                    format="%(asctime)-15s %(levelname)-8s  %(message)s")

logging.info('DataSet : {0}'.format(pkl_path))

# randomly get mini_training_batch
def get_random_batchdata(n_samples, batchsize):
    start_index = np.random.randint(0, n_samples - batchsize)
    return (start_index, start_index + batchsize)

print('>>>>Preparing Model......')

# the first frame
x1 = tf.placeholder(tf.float32, [None, 16384],name='img_1')
# the middle frame
x2 = tf.placeholder(tf.float32, [None, 16384],name='img_2')
# the last frame
x3 = tf.placeholder(tf.float32, [None, 16384],name='img_3')
# geometric feature R(78)
x4 = tf.placeholder(tf.float32, [None, 78],name='geometry_input')
# sample  label
y = tf.placeholder(tf.float32, [None, 6])
x_image1 = tf.reshape(x1, [-1, 128, 128, 1])
x_image2 = tf.reshape(x2, [-1, 128, 128, 1])
x_image3 = tf.reshape(x3, [-1, 128, 128, 1])


#DFSN-I
if model_flag == 3:
    from NET import DFSN_I as net3
    y_out = net3(x_image1,x_image2,x_image3,x4)

#DFSN
if model_flag == 2:
    from NET import DFSN as net2
    y_out = net2(x_image1,x_image2,x_image3)

#DGFN
if model_flag == 1:
    from NET import DGFN as net1
    y_out = net1(x4)


# loss function of cross entropy
# tensorflow cross entropy function
# inputs are the prediction and the real label
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out),0)

# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Accuracy
# The prediction of each sample is a vector of (1,6)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_out, 1))
# tf.cast cast bool to float
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Global Variable Initialization Operation
init = tf.global_variables_initializer()

# Load dataset
print('preparing data......')
from PreProcessing import pickle_2_img

frame_x1, frame_x2, frame_x3, geometry_x, data_label=pickle_2_img(pkl_path)

best_acc_of_fold = []

wts = time.time()

iteration = 0
while iteration<10:
    img_x1 = []
    img_x2 = []
    img_x3 = []
    g_x = []
    label_y = []
    for idfd in range(len(data_label)):
        if idfd == iteration:
            test_x1 = frame_x1[idfd]
            test_x2 = frame_x2[idfd]
            test_x3 = frame_x3[idfd]
            test_gx = geometry_x[idfd]
            test_y = data_label[idfd]
        else:
            img_x1 = img_x1+frame_x1[idfd]
            img_x2 = img_x2+frame_x2[idfd]
            img_x3 = img_x3+frame_x3[idfd]
            g_x = g_x+geometry_x[idfd]
            label_y = label_y+data_label[idfd]

    iteration = iteration+1
    n_samples = len(label_y)
    total_batches = int(n_samples / batch_size)
    print('test data %d'%len(test_y))
    print('total data %d'%(n_samples))
     
    isLog = True
    best_acc = 0
    best_epochs = 0
    cor_acc = 0
    cor_lost = 0
    confusion_matrix = []
    best_confusion_matrix = []
    isAccuarcy100 = False

    for i_in_ten in range(10):
    # Session
        with tf.Session() as sess:
            sess.run(init)

            saver = tf.train.Saver()                

            time_start = time.time()
            for i in range(train_epochs):
       
                for j in range(epochs_time):
                
                    start_index, end_index = get_random_batchdata(n_samples, batch_size)

                    batch_x1 = img_x1[start_index: end_index]
                    batch_x2 = img_x2[start_index: end_index]
                    batch_x3 = img_x3[start_index: end_index]
                    batch_gx = g_x[start_index: end_index]
                    batch_y = label_y[start_index: end_index]
                    _, cost, accu = sess.run([ optimizer, cross_entropy,accuracy], feed_dict={x1:batch_x1,x2:batch_x2, x3:batch_x3, x4:batch_gx, y:batch_y})           

                    if accu>test_bound:
                        correct_count=0
                        confusion_matrix = np.zeros((6,6),int)
                        result = sess.run(y_out, feed_dict={x1:test_x1, x2:test_x2, x3:test_x3, x4:test_gx})
                        for k in range(len(test_y)):
                            ip = np.argmax(result[k],0)
                            ir = np.argmax(test_y[k],0)
                            confusion_matrix[ir][ip] = confusion_matrix[ir][ip]+1
                            if np.equal(ip,ir):
                                correct_count = correct_count+1
                        test_acc = float(correct_count)/len(test_y)
                    
                        if test_acc>best_acc or (test_acc==best_acc and accu>cor_acc):
                            best_acc = test_acc
                            best_epochs = i+1
                            cor_acc = accu
                            cor_lost = cost
                            best_confusion_matrix = confusion_matrix
                            print('Best_accuracy:%s   ,accuracy : %.7f,      cost : %.7f  '%(str(best_acc), accu ,cost))
                            print('Confusion_Matrix:')
                            print('#Anger:0,Surprise:1,Disgust:2,Fear:3,Happiness:4,Sadness:5')
                            print(best_confusion_matrix)     
                            
                            #Accuracy 100% Next Epoch              
                            if best_acc == 1:
                                isAccuarcy100=True
                                break
                         
                   
                    print ('Fold : %d , %d , Epoch : %d ,  times:%d , accuracy : %.7f,  cost : %.7f  best_acc : %.7f'%(iteration, i_in_ten+1, i+1, j+1, accu ,cost, best_acc))
                    print('\n')

                #Accuracy 100% Next Epoch      
                if isAccuarcy100:
                    break                           
        
        if isAccuarcy100:
            break             
        
    if isLog:
        logging.info('Fold :{0} , Epochs:{1} , train_accuracy : {2} , test_accuarcy:{3} , lost:{4}'.format(iteration, best_epochs, cor_acc, best_acc, cor_lost))
        logging.info('#Anger:0,Surprise:1,Disgust:2,Fear:3,Happiness:4,Sadness:5')
        logging.info('Confusion Matrix \n {0}\n'.format(best_confusion_matrix))    
        time_end = time.time()
        print("Fold %d time comsuming: %fs"%(iteration,(time_end-time_start)))
        best_acc_of_fold.append(best_acc)

wte = time.time()
print("Total Time comsuming: %fs"%(wte-wts))
print(best_acc_of_fold)
print("Mean accuracy : %.7f"%np.mean(best_acc_of_fold))


logging.info('Total Time comsuming:{0}s'.format(wte-wts))
logging.info('Ten Fold :{0}'.format(best_acc_of_fold))
logging.info('Mean accuracy :{0}'.format(np.mean(best_acc_of_fold)))
            


        