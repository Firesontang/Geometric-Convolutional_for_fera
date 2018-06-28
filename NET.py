# -*- coding: utf-8 -*-
# @Author: Yan Tang
# @Date:   2018-06-27 

'''
----------------------------------------------------------------
Network Construction of :
DGFN
DFSN
DFSN-I
----------------------------------------------------------------
'''

import tflearn as tfl
import tensorflow as tf

#DFSN-I
def DFSN_I(img1,img2,img3,geo):

    conv1_fm = 12
    conv2_fm = 24
    conv3_fm = 36
    conv4_fm = 48
    conv5_fm = 64

    conv1_size = 3
    conv2_size = 5
    conv3_size = 7
    conv4_size = 7
    conv5_size = 3

    #FE layers for img1
    net1 = tfl.conv_2d(img1, conv1_fm, conv1_size, activation='relu', name='n1_conv1_1')
    net1 = tfl.conv_2d(net1, conv1_fm, conv1_size, activation='relu', name='n1_conv1_2')
    net1 = tfl.max_pool_2d(net1, 2, 2, name='n1_pool1')
    net1 = tfl.conv_2d(net1, conv2_fm, conv2_size, activation='relu', name='n1_conv2_1')
    net1 = tfl.conv_2d(net1, conv2_fm, conv2_size, activation='relu', name='n1_conv2_2')
    net1 = tfl.max_pool_2d(net1, 2, 2, name='n1_pool2')
    net1 = tfl.conv_2d(net1, conv3_fm, conv3_size, activation='relu', name='n1_conv3_1')
    net1 = tfl.max_pool_2d(net1, 2, 2, name='n1_pool3')
    net1 = tfl.conv_2d(net1, conv4_fm, conv4_size, activation='relu', name='n1_conv4_1')
    net1 = tfl.max_pool_2d(net1, 2, 2, name='n1_pool4')
    net1 = tfl.conv_2d(net1, conv5_fm, conv5_size, activation='relu', name='n1_conv5_1')
    net1 = tfl.max_pool_2d(net1, 2, 2, name='n1_pool5')
    net1 = tf.reshape(net1, [-1, conv5_fm*4*4],name='n1_re')

    #FE layers for img2
    net2 = tfl.conv_2d(img2, conv1_fm, conv1_size, activation='relu', name='n2_conv1_1')
    net2 = tfl.conv_2d(net2, conv1_fm, conv1_size, activation='relu', name='n2_conv1_2')
    net2 = tfl.max_pool_2d(net2, 2, 2, name='n2_pool1')
    net2 = tfl.conv_2d(net2, conv2_fm, conv2_size, activation='relu', name='n2_conv2_1')
    net2 = tfl.conv_2d(net2, conv2_fm, conv2_size, activation='relu', name='n2_conv2_2')
    net2 = tfl.max_pool_2d(net2, 2, 2, name='n2_pool2')
    net2 = tfl.conv_2d(net2, conv3_fm, conv3_size, activation='relu', name='n2_conv3_1')
    net2 = tfl.max_pool_2d(net2, 2, 2, name='n2_pool3')
    net2 = tfl.conv_2d(net2, conv4_fm, conv4_size, activation='relu', name='n2_conv4_1')
    net2 = tfl.max_pool_2d(net2, 2, 2, name='n2_pool4')
    net2 = tfl.conv_2d(net2, conv5_fm, conv5_size, activation='relu', name='n2_conv5_1')
    net2 = tfl.max_pool_2d(net2, 2, 2, name='n2_pool5')
    net2 = tf.reshape(net2, [-1, conv5_fm*4*4],name='n2_re')

    #FE layers for img3
    net3 = tfl.conv_2d(img3, conv1_fm, conv1_size, activation='relu', name='n3_conv1_1')
    net3 = tfl.conv_2d(net3, conv1_fm, conv1_size, activation='relu', name='n3_conv1_2')
    net3 = tfl.max_pool_2d(net3, 2, 2, name='n3_pool1')
    net3 = tfl.conv_2d(net3, conv2_fm, conv2_size, activation='relu', name='n3_conv2_1')
    net3 = tfl.conv_2d(net3, conv2_fm, conv2_size, activation='relu', name='n3_conv2_2')
    net3 = tfl.max_pool_2d(net3, 2, 2, name='n3_pool2')
    net3 = tfl.conv_2d(net3, conv3_fm, conv3_size, activation='relu', name='n3_conv3_1')
    net3 = tfl.max_pool_2d(net3, 2, 2, name='n3_pool3')
    net3 = tfl.conv_2d(net3, conv4_fm, conv4_size, activation='relu', name='n3_conv4_1')
    net3 = tfl.max_pool_2d(net3, 2, 2, name='n3_pool4')
    net3 = tfl.conv_2d(net3, conv5_fm, conv5_size, activation='relu', name='n3_conv5_1')
    net3 = tfl.max_pool_2d(net3, 2, 2, name='n3_pool5')
    net3 = tf.reshape(net3, [-1, conv5_fm*4*4],name='n3_re')

    #geometric feature R(78)
    p3 = tf.reshape(geo, [-1, 78], name='n4_re')
    
    fc_net1 = tfl.fully_connected(net1, 512, activation='tanh', name='fc1')
    fc_net2 = tfl.fully_connected(net2, 512, activation='tanh', name='fc2')
    fc_net3 = tfl.fully_connected(net3, 512, activation='tanh', name='fc3')
    fc_net4 = tfl.fully_connected( p3  , 512, activation='tanh', name='fc4')

    fc_net=tf.concat([fc_net1, fc_net2, fc_net3, fc_net4], 1, name='fusion_1')

    fc_net = tfl.fully_connected(fc_net, 1024, activation='tanh', name='fc5')
    fc_net = tfl.dropout(fc_net, 0.3, name='drop1')
    softmax = tfl.fully_connected(fc_net, 6,activation='softmax', name='prob')
    
    return softmax

#DFSN
def DFSN(img1,img2,img3):

    conv1_fm = 12
    conv2_fm = 24
    conv3_fm = 36
    conv4_fm = 48
    conv5_fm = 64

    conv1_size = 3
    conv2_size = 5
    conv3_size = 7
    conv4_size = 7
    conv5_size = 3

    #FE layers for img1
    net1 = tfl.conv_2d(img1, conv1_fm, conv1_size, activation='relu', name='n1_conv1_1')
    net1 = tfl.conv_2d(net1, conv1_fm, conv1_size, activation='relu', name='n1_conv1_2')
    net1 = tfl.max_pool_2d(net1, 2, 2, name='n1_pool1')
    net1 = tfl.conv_2d(net1, conv2_fm, conv2_size, activation='relu', name='n1_conv2_1')
    net1 = tfl.conv_2d(net1, conv2_fm, conv2_size, activation='relu', name='n1_conv2_2')
    net1 = tfl.max_pool_2d(net1, 2, 2, name='n1_pool2')
    net1 = tfl.conv_2d(net1, conv3_fm, conv3_size, activation='relu', name='n1_conv3_1')
    net1 = tfl.max_pool_2d(net1, 2, 2, name='n1_pool3')
    net1 = tfl.conv_2d(net1, conv4_fm, conv4_size, activation='relu', name='n1_conv4_1')
    net1 = tfl.max_pool_2d(net1, 2, 2, name='n1_pool4')
    net1 = tfl.conv_2d(net1, conv5_fm, conv5_size, activation='relu', name='n1_conv5_1')
    net1 = tfl.max_pool_2d(net1, 2, 2, name='n1_pool5')
    net1 = tf.reshape(net1, [-1, conv5_fm*4*4],name='n1_re')

    #FE layers for img2
    net2 = tfl.conv_2d(img2, conv1_fm, conv1_size, activation='relu', name='n2_conv1_1')
    net2 = tfl.conv_2d(net2, conv1_fm, conv1_size, activation='relu', name='n2_conv1_2')
    net2 = tfl.max_pool_2d(net2, 2, 2, name='n2_pool1')
    net2 = tfl.conv_2d(net2, conv2_fm, conv2_size, activation='relu', name='n2_conv2_1')
    net2 = tfl.conv_2d(net2, conv2_fm, conv2_size, activation='relu', name='n2_conv2_2')
    net2 = tfl.max_pool_2d(net2, 2, 2, name='n2_pool2')
    net2 = tfl.conv_2d(net2, conv3_fm, conv3_size, activation='relu', name='n2_conv3_1')
    net2 = tfl.max_pool_2d(net2, 2, 2, name='n2_pool3')
    net2 = tfl.conv_2d(net2, conv4_fm, conv4_size, activation='relu', name='n2_conv4_1')
    net2 = tfl.max_pool_2d(net2, 2, 2, name='n2_pool4')
    net2 = tfl.conv_2d(net2, conv5_fm, conv5_size, activation='relu', name='n2_conv5_1')
    net2 = tfl.max_pool_2d(net2, 2, 2, name='n2_pool5')
    net2 = tf.reshape(net2, [-1, conv5_fm*4*4],name='n2_re')

    #FE layers for img3
    net3 = tfl.conv_2d(img3, conv1_fm, conv1_size, activation='relu', name='n3_conv1_1')
    net3 = tfl.conv_2d(net3, conv1_fm, conv1_size, activation='relu', name='n3_conv1_2')
    net3 = tfl.max_pool_2d(net3, 2, 2, name='n3_pool1')
    net3 = tfl.conv_2d(net3, conv2_fm, conv2_size, activation='relu', name='n3_conv2_1')
    net3 = tfl.conv_2d(net3, conv2_fm, conv2_size, activation='relu', name='n3_conv2_2')
    net3 = tfl.max_pool_2d(net3, 2, 2, name='n3_pool2')
    net3 = tfl.conv_2d(net3, conv3_fm, conv3_size, activation='relu', name='n3_conv3_1')
    net3 = tfl.max_pool_2d(net3, 2, 2, name='n3_pool3')
    net3 = tfl.conv_2d(net3, conv4_fm, conv4_size, activation='relu', name='n3_conv4_1')
    net3 = tfl.max_pool_2d(net3, 2, 2, name='n3_pool4')
    net3 = tfl.conv_2d(net3, conv5_fm, conv5_size, activation='relu', name='n3_conv5_1')
    net3 = tfl.max_pool_2d(net3, 2, 2, name='n3_pool5')
    net3 = tf.reshape(net3, [-1, conv5_fm*4*4],name='n3_re')
    
    fc_net1 = tfl.fully_connected(vgg_net1, 512, activation='tanh', name='fc1')
    fc_net2 = tfl.fully_connected(vgg_net2, 512, activation='tanh', name='fc2')
    fc_net3 = tfl.fully_connected(vgg_net3, 512, activation='tanh', name='fc3')

    fc_net=tf.concat([net1, net2, net3], 1, name='fusion_1')

    fc_net = tfl.fully_connected(fc_net, 1024, activation='tanh', name='fc5')
    fc_net = tfl.dropout(fc_net, 0.3, name='drop1')
    softmax = tfl.fully_connected(fc_net, 6,activation='softmax', name='prob')

    return softmax

#DGFN
def DGFN(geo):
    p3 = tf.reshape(geo, [-1, 78], name='geo_in1')
    fc_net = tfl.fully_connected(p3, 512, activation='tanh', name='fc1')
    fc_net = tfl.dropout(fc_net, 0.3, name='drop1')
    softmax = tfl.fully_connected(fc_net, 6,activation='softmax', name='prob')

    return softmax