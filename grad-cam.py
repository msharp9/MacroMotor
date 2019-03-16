# -*- coding: utf-8 -*-
"""
Created on Wed Mar 06 09:08:20 2019

@author: ramachandran
edited
"""

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D

import numpy as np
# from matplotlib import pyplot as plt
from keras import models, layers

import tensorflow as tf
import cv2,sys,os

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.InteractiveSession(config=config)

def build_model(state_size, action_size):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                     input_shape=state_size))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_size)) # action_size
    model.summary()
    return model

model = build_model((176, 200, 4), 15)
model.load_weights("model/ddqn_2000.h5")

# model = load_model("model/ddqn_2000.h5")
print (model.summary())

img_path = "img/1.png"
last_conv2d_layername = 'conv2d_3'

img = cv2.imread(img_path,0)
print(img.shape)
history = np.stack((img, img, img, img), axis=2)
# print(history.shape)
img_tensor = np.reshape([history], (1, 176, 200, 4))
# print(img_tensor.shape)

#
# img_tensor = image.img_to_array(history)
# img_tensor = np.expand_dims(history,axis=0)
# print(img.shape)
# print(img, img_tensor)
#
preds = model.predict(img_tensor)
# print(preds)
index = np.argmax(preds[0])
# print (index)
#
model_output = model.output[:,index]
# print (model_output)
#
last_conv_layer = model.get_layer(last_conv2d_layername)
# print (last_conv_layer)
#
grads = K.gradients(model_output,last_conv_layer.output)[0]
# print(grads)

pooled_grads = K.mean(grads, axis=(0,1,2))
# print(pooled_grads)

iterate = K.function([model.input,K.learning_phase()],[pooled_grads,last_conv_layer.output[0]])
# print(iterate)

pooled_grads_value, conv_layer_output_value = iterate([img_tensor,0])

# print(pooled_grads_value, conv_layer_output_value)
# print (pooled_grads_value.shape)
# print (conv_layer_output_value.shape)
#
iters = len(pooled_grads_value)
# print (iters)

for i in range(iters):
    conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

# print(conv_layer_output_value)

heatmap = np.mean(conv_layer_output_value, axis=-1)
# print(heatmap)
# print(np.absolute(heatmap))
# heatmap = np.absolute(heatmap)
heatmap = np.maximum(heatmap,0)
# print(heatmap)
# print(np.absolute(heatmap))
# print(np.max(heatmap))
heatmap /= np.max(np.absolute(heatmap))
# print(heatmap)
# plt.matshow(heatmap)
# print (heatmap.shape)
# print (np.sum(heatmap))

img = cv2.imread(img_path)

print((img.shape[1],img.shape[0]))
print(heatmap.shape)

heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
# # heatmap = cv2.resize(heatmap,(0,0),fx=(296.0/67.0),fy=(296.0/67.0))
heatmap = np.uint8(255*heatmap)
# print (heatmap.shape)
# print (np.sum(heatmap))
# print (type(heatmap))
# print (np.max(heatmap))
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# print (heatmap.shape)
# print (np.sum(heatmap))
# #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
# #img = cv2.resize(img,(0,0),fx=296.0/300.0,fy=296/300.0)
# print (np.sum(img))
print(img.shape)
superimposed_img = heatmap*0.4 + img
cv2.imwrite("img/1_cam.jpg", superimposed_img)
