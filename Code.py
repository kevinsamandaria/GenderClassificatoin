# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 20:08:24 2021

@author: USER
"""

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def import_data():
    dataset = pd.read_csv("gender_classification.csv")
    x = dataset[["long_hair", "forehead_width_cm", "forehead_height_cm", "nose_wide", "nose_long","lips_thin", "distance_nose_to_lip_long"]]
    y = dataset[["gender"]]
    
    return x,y

def preprocess_data(features, target):
    features = MinMaxScaler().fit_transform(features)
    target = OneHotEncoder(sparse=False).fit_transform(target)
    
    return features, target

layers = {
    'input': 7,
    'hidden': 7,
    'output': 2
    }

weight = {
    'hidden': tf.Variable(tf.random.normal([layers['input'], layers['hidden']])),
    'output': tf.Variable(tf.random.normal([layers['hidden'], layers['output']]))
    }

bias = {
    'hidden': tf.Variable(tf.random.normal([layers['hidden']])),
    'output': tf.Variable(tf.random.normal([layers['output']]))
    }

def activate(x):
    return tf.nn.sigmoid(x)

def foward_pass(features):
    x1 = tf.matmul(features, weight['hidden']) + bias['hidden']
    y1 = activate(x1)
    
    x2 = tf.matmul(y1, weight['output']) + bias['output']
    y2 = activate(x2)
    
    return y2

features_temp = tf.placeholder(tf.float32, [None, layers['hidden']])
target_temp = tf.placeholder(tf.float32, [None, layers['output']])

output = foward_pass(features_temp)
error = tf.reduce_mean(0.5 * (target_temp - output) ** 2)

Learning_rate = 0.1
epoch = 5000 
training = tf.train.GradientDescentOptimizer(0.1).minimize(error)

features, target = import_data()
features, target = preprocess_data(features, target)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(epoch+1):
        train_data = {
            features_temp : features_train,
            target_temp : target_train
            }
        sess.run(training, feed_dict = train_data)
        curr_error = sess.run(error, feed_dict = train_data)
        
        if i % 200 == 0:
            print(f"epoch: {i}, Current Error = {curr_error}")
            
    accuracy = tf.equal(tf.argmax(target_temp, axis = 1), tf.argmax(output, axis = 1))
    test_data = {
        features_temp : features_test,
        target_temp: target_test
        }
    result = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    print(f"Accuracy = {sess.run(result, feed_dict = test_data) * 100}%")