#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install split-folders


# In[2]:


import splitfolders


# In[3]:


input_folder=r'F:\ANURAG GUPTA\MastitisDataForSahiwal'


# In[4]:


splitfolders.ratio(input_folder, output=r"F:\ANURAG GUPTA\cell_images2",
                  seed=42, ratio=(.7, .2, .1),
                  group_prefix=None)


# In[11]:


pip install keras-unet


# In[21]:


import keras_unet.models
#from unet_model_with_functions_of_blocks import build_unet
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
#from patchify import patchify, unpatchify
import tifffile as tiff
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam


# In[22]:


seed=24
batch_size=8
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[23]:


img_data_gen_args = dict(rescale = 1/255.,
                         rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')


# In[24]:


mask_data_gen_args = dict(rescale = 1/255.,  #Original pixel values are 0 and 255. So rescaling to 0 to 1
                        rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect',
                      preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype))


# In[26]:


image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_generator = image_data_generator.flow_from_directory(r"F:\ANURAG GUPTA\cell_images2\train", 
                                                           seed=seed, 
                                                           batch_size=batch_size,
                                                           class_mode=None)


# In[ ]:


mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_generator = mask_data_generator.flow_from_directory("data2/train_masks/", 
                                                         seed=seed, 
                                                         batch_size=batch_size,
                                                         color_mode = 'grayscale',   #Read masks in grayscale
                                                         class_mode=None)


# In[27]:


valid_img_generator = image_data_generator.flow_from_directory(r"F:\ANURAG GUPTA\cell_images2\val", 
                                                               seed=seed, 
                                                               batch_size=batch_size, 
                                                               class_mode=None)


# In[ ]:


valid_mask_generator = mask_data_generator.flow_from_directory("data2/val_masks/", 
                                                               seed=seed, 
                                                               batch_size=batch_size, 
                                                               color_mode = 'grayscale',   #Read masks in grayscale
                                                               class_mode=None)


# In[29]:


train_generator = zip(image_generator)
val_generator = zip(valid_img_generator)


# In[28]:



train_generator = zip(image_generator, mask_generator)
val_generator = zip(valid_img_generator, valid_mask_generator)


# In[ ]:





# In[30]:


x = image_generator.next()
y = mask_generator.next()
for i in range(0,1):
    image = x[i]
    mask = y[i]
    plt.subplot(1,2,1)
    plt.imshow(image[:,:,0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(mask[:,:,0])
    plt.show()


# In[ ]:




