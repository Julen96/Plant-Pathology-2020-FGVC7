#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt


# In[2]:


def load_images(train):
    
    # This function loads the images, resizes them and puts them into an array
    
    img_size = 448
    train_image = []
    counter = 0
    for name in train['image_id']:
        counter += 1
        if (counter % 100 == 0):
            print('we have loaded', counter , 'images')
        path = directory + 'images/' + name + '.jpg'
        img = cv2.imread(path)
        image = cv2.resize(img, (img_size, img_size))
        train_image.append(image)
    train_image_array = np.array(train_image)
    
    return train_image_array


# In[3]:


def show_distribution():
    n_healthy = sum(df_train['healthy'])
    n_multiple = sum(df_train['multiple_diseases'])
    n_rust = sum(df_train['rust'])
    n_scab = sum(df_train['scab'])

    print('Number of Healthy images:', n_healthy)
    print('Number of Multiple_diseases images:', n_multiple)
    print('Number of Rust images:', n_rust)
    print('Number of Scab images:', n_scab)
    
    distribution_dictionary = {'healthy' : n_healthy,
                               'multiple_diseases' : n_multiple,
                               'rust' : n_rust,
                               'scab' : n_scab}
    
    return distribution_dictionary


# In[4]:


def append_dataframe(df_train, name, train_add):
    # This functions append to the dataframe the new image's labels

    max_len = len(df_train)

    for i in range(0, len(train_add)):
        data = {'image_id' : 'Train_' + str(max_len + i), 
                'healthy' : [0],
                'multiple_diseases' : [0],
                'rust' : [0],
                'scab' : [0]}
        df_new = pd.DataFrame(data)
        df_new[name] = 1
        df_train = df_train.append(df_new)
        
    return df_train


# In[5]:


def append_images(name, add_images):
    
    train_add = train_img.copy()
    
    if name == 'rust': # Don't add anything
        print('rust')

    elif name == 'multiple_diseases': # Add 4 times the amount of images and more
        for i in range(0,4):
            train_add = np.concatenate((train_add, train_img))
        train_add = np.concatenate((train_add, train_img[0:76]))
        
    else: # Add the needed amount.
        train_add = train_img[0:add_images]
        
    return train_add


# In[6]:


def write_images(train_img, len_previous):
    
    for i in range(0, len(train_img)):
        img_name = 'Train_' + str(len_previous + i) + '.jpg'
        cv2.imwrite(directory + 'images/' + img_name, train_img[i])


# In[7]:


directory = ''
df_train = pd.read_csv(directory + 'train.csv')


# ## Multiple_diseases augmentation

# In[9]:


names = ['healthy', 'multiple_diseases', 'rust', 'scab']
distr_dict = show_distribution()
print()

for name in names:
    if name == 'rust':
        print('Russt!')
        
    else:
        df_len_previous = len(df_train)

        train_img = load_images(df_train)
        print()
        train_img = train_img[df_train[name] == 1] # Takes the images of the label
        n_samples = distr_dict[name] # Takes the number of samples of that images

        add_images = distr_dict['rust'] - n_samples # The amount of images we have to add
        print('images we have to add:', add_images)

        train_add = append_images(name, add_images)

        df_train = append_dataframe(df_train, name, train_add)

        write_images(train_add, df_len_previous)
    
df_train


# In[10]:


df_train.to_csv(directory + 'train_modified.csv')

