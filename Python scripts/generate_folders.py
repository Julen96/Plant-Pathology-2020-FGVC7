#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt


# In[2]:


def add_status(train):
    
    # It has been done this wah in order to avoid unnecessary warnings
    train['status'] = 0
    train['label'] = 0
    
    train.loc[train.healthy == 1, 'status'] = 'healthy'
    train.loc[train.healthy == 1, 'label'] = 0
    train.loc[train.multiple_diseases == 1, 'status'] = 'multiple_diseases'
    train.loc[train.multiple_diseases == 1, 'label'] = 1
    train.loc[train.rust == 1, 'status'] = 'rust'
    train.loc[train.rust == 1, 'label'] = 2
    train.loc[train.scab == 1, 'status'] = 'scab'
    train.loc[train.scab == 1, 'label'] = 3
    
    return train


# In[3]:


def load_images(train, directory):
    
    # This function loads the images, resizes them and puts them into an array
    
    img_size = 720
    train_image = []
    for name in train['image_id']:
        path = directory + 'images/' + name + '.jpg'
        img = cv2.imread(path)
        image = cv2.resize(img, (img_size, img_size))
        train_image.append(image)
    train_image_array = np.array(train_image)
    
    return train_image_array


# In[4]:


def save_images(folder_name, x, y):
    healthy_count = 0
    multiple_diseases_count = 0
    rust_count = 0
    scab_count = 0

    for i in range(0, len(x)):
        if y[i] == 0:
            healthy_count += 1
            name = 'healthy_' + str(healthy_count) + '.jpg'
            cv2.imwrite(folder_name + 'healthy/' + name, x[i])
        elif y[i] == 1:
            multiple_diseases_count +=1
            name = 'multiple_diseases_' + str(multiple_diseases_count) + '.jpg'
            cv2.imwrite(folder_name + 'multiple_diseases/' + name, x[i])
        elif y[i] == 2:
            rust_count +=1
            name = 'rust_' + str(rust_count) + '.jpg'
            cv2.imwrite(folder_name + 'rust/' + name, x[i])
        elif y[i] == 3:
            scab_count +=1
            name = 'scab_' + str(scab_count) + '.jpg'
            cv2.imwrite(folder_name + 'scab/' + name, x[i])


# In[5]:


def make_folders(directory):
    import os
    classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
    os.mkdir(directory + 'image_generator')
    os.mkdir(directory + 'image_generator/train')
    for cls in classes:
        os.mkdir(directory + 'image_generator/train/' + cls )
    os.mkdir(directory + 'image_generator/validation')
    for cls in classes:
        os.mkdir(directory + 'image_generator/validation/' + cls )


# In[6]:


directory = ''#'C:/Users/julen/OneDrive/Escritorio/IA/CS577-Deep-Learning/Project/'
df_train = pd.read_csv(directory + 'train_modified.csv')
df_train = add_status(df_train)
train_img = load_images(df_train, directory)
make_folders(directory)


# In[7]:


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(train_img, df_train['label'].to_numpy(), 
                                                  stratify = df_train['label'].to_numpy(), test_size = 0.2)


# In[8]:


save_images(directory + 'image_generator/train/', x_train, y_train)
save_images(directory + 'image_generator/validation/', x_val, y_val)


# In[ ]:




