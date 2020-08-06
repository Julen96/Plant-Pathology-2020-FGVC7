#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras import models, Sequential
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, Dropout, Multiply
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
import time
import numpy as np
from keras.applications.resnet import ResNet50
import tensorflow as tf
from keras import losses
import keras.backend as K
import tensorflow as tf
from keras.models import model_from_json 




def load_generators_augmented(img_size = 448):
    # This function loads the generator for the train and validation sets.
    # This first approach takes the test images from the disk
    
    from keras.preprocessing.image import ImageDataGenerator

    img_size = 448
    train_dir = './image_generator/train'
    val_dir = './image_generator/validation'
    batch = 8


    train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 30,
                                       width_shift_range = 0.2, height_shift_range = 0.2, 
                                       shear_range = 0.2, zoom_range = 0.2, 
                                       horizontal_flip = True, vertical_flip = True,
                                       fill_mode = 'nearest')


    validation_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, 
                                                        target_size = (img_size,img_size),
                                                        batch_size = batch,
                                                        class_mode = 'categorical')

    validation_generator = train_datagen.flow_from_directory(val_dir, 
                                                        target_size = (img_size,img_size),
                                                        batch_size = batch,
                                                        class_mode = 'categorical')
    #test_generator = test_datagen.flow_from_directory(test_dir, 
    #                                                   target_size = (img_size,img_size),
    #                                                    batch_size = batch,
    #                                                    class_mode = 'categorical')
    
    return(train_generator, validation_generator)


def load_images(df, directory):
    
    # This function loads the images, resizes them and puts them into an array
    img_size = 224
    train_image = []
    for name in df['image_id']:
        path = directory + 'images/' + name + '.jpg'
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        train_image.append(img)
    train_image_array = np.array(train_image)
    
    return train_image_array


def load_generators_augmented_flow():
    # This function loads the generator for the test.
    # This second approach takes the test images from the memory
    
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.image import ImageDataGenerator
    
    directory = ''
    df_train = pd.read_csv(directory + 'train.csv')
    x_train = load_images(df_train, directory)

    y_train = df_train[['healthy', 'multiple_diseases', 'rust', 'scab']].to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 2020,
                                                      shuffle = True, stratify = y_train)
                                                      
    train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 30,
                                           width_shift_range = 0.2, height_shift_range = 0.2, 
                                           shear_range = 0.2, zoom_range = 0.2, 
                                           horizontal_flip = True, vertical_flip = True,
                                           fill_mode = 'nearest')


    validation_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow(x_train, y_train, batch_size = 4)
    validation_generator = validation_datagen.flow(x_val, y_val, batch_size = 4)
    
    return (train_generator, validation_generator)
    

from keras import backend as K
from keras.layers import Layer, InputSpec


class GlobalKMaxPooling2D(Layer): #Inherits the properties of Layer class
    """K Max Pooling operation for spatial data.
    
    # Arguments
        
        data_format: A string,
            one of `"channels_last"` (default) or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be `"channels_last"`.
            
        K: An Integer,
            states the number of selected maximal values over which the
            average is going to be computed.
            
            
    # Input shape
    
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
            
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
            
    # Output shape
    
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, pooled_rows, pooled_cols, channels)`
            
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, pooled_rows, pooled_cols)`    
    
    
    """
    

    def __init__(self, data_format=None, k = 10, **kwargs):
        super(GlobalKMaxPooling2D, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        self.k = k

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], input_shape[3])
        else:
            return (input_shape[0], input_shape[1])

    def get_config(self):
        config = {'data_format': self.data_format, 'k' : self.k}
        base_config = super(GlobalKMaxPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
        
    def call(self, inputs):
        if self.data_format == 'channels_last':
            # Here first sort
            # Then take K maximum values
            # Then average them
            k = self.k

            input_reshaped = tf.reshape(inputs, [tf.shape(inputs)[0], -1, tf.shape(inputs)[3]])
            input_reshaped = tf.reshape(input_reshaped, [tf.shape(input_reshaped)[0], tf.shape(input_reshaped)[2], tf.shape(input_reshaped)[1]])
            top_k = tf.math.top_k(input_reshaped, k=k, sorted = True, name = None)[0]
            mean = tf.keras.backend.mean(top_k, axis = 2)
            #assert ((input_reshaped.get_shape()[0], input_reshaped.get_shape()[-1]) == mean.get_shape())
        
        return mean


def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=5, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    # This function makes the learning rate schedule.
    
    lr_max = lr_max #* strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    return lrfn


lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

from keras.callbacks import ModelCheckpoint
filepath="Elope_xception_V2_best_model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_AUC', verbose=1, save_best_only=True, mode='max')



# In[10]:


#!pip install -q efficientnet
# from keras.applications.resnet import ResNet50
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.efficientnet import EfficientNetB7
# import efficientnet.tfkeras as efn
from keras.losses import categorical_crossentropy
from keras import metrics
import keras
from keras import models, Sequential
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, Dropout, Multiply
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Reshape
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
import time
from keras.applications.xception import Xception


    
class ElopeModel(object):
    
    def __init__(self):
        
        # Model
        self.model = self.build_model()
        # Print a model summary
        self.model.summary()
        
        # Optimizer
        self.optimizer = 'Adam'
        
        
        self.loss_parameters = {'means' : 0, 'lr' : tf.constant(0.5,tf.float32), 'landa' : tf.constant(0.0003 ,tf.float32),
                               'gamma' : tf.constant(16.0,tf.float32), 'm' : tf.constant(0.75,tf.float32)}

        
        #Loss Function
        self.loss_func = self.model_loss()
        
        self.compile()
        
        
    def build_model(self):
        
        input_shape = (448, 448, 3)
        # model_resnet = ResNet50(include_top = False, weights = 'imagenet', input_shape = input_shape)
        model_xception = Xception(include_top = False, weights = 'imagenet', input_shape = input_shape)
        # model_resnet.trainable = False
        #model_efficientnet = efn.EfficientNetB7(input_shape = input_shape, weights = 'imagenet', include_top = False)
        #model_inception = InceptionResNetV2(include_top = False, weights = 'imagenet', input_shape = input_shape)
        model = Sequential()
        
        model.add(model_xception)
        model.add(GlobalKMaxPooling2D(data_format = 'channels_last' , k = 4))
        # model.add(Dense(128, activation = 'relu'))
        # model.add(Dense(64, activation = 'relu'))
        model.add(Dense(4, activation = 'softmax'))
        # tf.compat.v1.enable_eager_execution()
        # print('Eager execution:', tf.executing_eagerly())
        
        return model
    
    
    def calculate_additional_loss(self, y_true, y_pred): #fx_tensor = Output of layer
       
       
        fx_tensor = self.model.layers[-2].output
       
       
        mean_tensor = self.loss_parameters['means']
        alpha = self.loss_parameters['lr']

        dimension_tensor = K.shape(fx_tensor)[1]
        num_classes_tensor = K.shape(y_true)[1]
        if isinstance(mean_tensor, int): # Initialize them if they are not yet initialized
            mean_tensor = tf.random.uniform([num_classes_tensor, dimension_tensor], dtype = tf.dtypes.float32)
        num_samples_tensor = K.shape(fx_tensor)[0]
        
        # Ensure they are float 32
        fx_tensor = tf.cast(fx_tensor, dtype = tf.dtypes.float32)
        

        fx_expanded = tf.broadcast_to(tf.expand_dims(fx_tensor, axis = -1), [num_samples_tensor, dimension_tensor, num_classes_tensor])
        y_true_expanded = tf.broadcast_to(tf.expand_dims(y_true, axis = 1), [num_samples_tensor, dimension_tensor, num_classes_tensor])
        mean_expanded = tf.broadcast_to(tf.expand_dims(mean_tensor, axis = 0), [num_samples_tensor, num_classes_tensor, dimension_tensor])

        mean_expanded = tf.transpose(mean_expanded, perm = [0,2,1])

        up = tf.reduce_sum(tf.multiply(tf.subtract(mean_expanded, fx_expanded), tf.cast(y_true_expanded, dtype = tf.dtypes.float32)), axis = 0)
        y_true_cut = tf.cast(tf.reduce_sum(y_true, axis = 0), dtype = tf.dtypes.float32)
        down = tf.add(y_true_cut, tf.constant(1, dtype = tf.dtypes.float32))
        delta = tf.divide(up, down)
        delta = tf.transpose(delta)

        mean_new = tf.subtract(mean_tensor, tf.scalar_mul(alpha, delta))

        # Now calculate the loss
        mean_new_expanded = tf.broadcast_to(tf.expand_dims(mean_new, axis = 0), [num_samples_tensor, num_classes_tensor, dimension_tensor])
        mean_new_expanded = tf.transpose(mean_new_expanded,  perm = [0,2,1]) 
        inside = tf.reduce_sum(tf.multiply(tf.subtract(fx_expanded, mean_new_expanded), tf.cast(y_true_expanded, dtype = tf.dtypes.float32)), axis = 2)


        tot = tf.reduce_sum(tf.multiply(inside, inside)) # Apply the norm
        down = tf.multiply(tf.constant(2), num_samples_tensor)
        loss_within_class = tf.divide(tot, tf.cast(down, dtype = tf.dtypes.float32))
        
        mean_tensor = mean_new 
        self.loss_parameters['means'] = mean_tensor
        
        
        # N = number of samples in the batch
        # f(xn) -> Dimension = 2048 (not the batch because it it xn, only 1 sample)
        # class means -> should be the same as f(xn) x number of classes
        # Therefore ||f(xn) - U(cn)||^2 will be a number, because f(xn) is a vector, not a matrix

        
        ###########################################
        ###Here we will calculate the other Loss###
        ###########################################
        
        gamma = self.loss_parameters['gamma']
        m = self.loss_parameters['m']
        # m -> Margin
        # P -> class-pairs in the current batch
        # |P| -> Cardinality of P, number of elements in the set
        
        # print(mean_tensor.numpy())
        

        dimension_tensor = K.shape(fx_tensor)[1]
        num_classes_tensor = K.shape(y_true)[1]
        num_samples_tensor = K.shape(y_true)[0]

        # First we have to expand the mean matrix in order to substract it
        mean_expanded = tf.broadcast_to(tf.expand_dims(mean_tensor, axis = 0), [num_classes_tensor, num_classes_tensor, dimension_tensor])
        # We transpose it to do the subtraction
        mean_trans = tf.transpose(mean_expanded, [1,0,2])

        # We subtract it and do the norm through the "dimensions" axis.
        norm = tf.subtract(mean_expanded, mean_trans)
        norm = tf.reduce_sum(tf.multiply(norm, norm), axis = 2) 

        # Finally we subtract to m the previously calculated norm vector
        m_tensor = tf.scalar_mul(m, tf.ones(K.shape(norm), tf.float32))
        inside = tf.subtract(m_tensor, tf.cast(norm, tf.float32))

        # Now we only take the lower part diagonal matrix, and as we cannot delete the diagonal, we subtract it in the final

        mat = tf.linalg.LinearOperatorLowerTriangular(inside)
        diagonal = mat.diag_part() # diagonal
        mat = mat.to_dense() # Lower triangular matrix with diagonal

        zeros = tf.zeros(K.shape(norm), tf.float32)
        maximum = tf.maximum(mat, zeros)
        max_squared = tf.square(maximum)

        summ = tf.reduce_sum(max_squared)

        diag_sum = tf.reduce_sum(tf.multiply(diagonal, diagonal))

        summ = tf.subtract(summ, diag_sum)



        counter = tf.ones([num_classes_tensor, num_classes_tensor], tf.float32)
        counter = tf.linalg.LinearOperatorLowerTriangular(counter)

        count_diag = counter.diag_part()
        count_triang = counter.to_dense()
        counter = tf.subtract(tf.reduce_sum(count_triang), tf.reduce_sum(count_diag))


        loss_between_class = tf.multiply(tf.divide(gamma, tf.multiply(tf.constant(4, tf.float32), counter)), summ)
        
        
        loss = tf.math.add(loss_between_class, loss_within_class)
        
        return loss
    
    
    
    
    def model_loss(self):
        """" Wrapper function which calculates auxiliary values for the complete loss function.
         Returns a *function* which calculates the complete loss given only the input and target output """
        # This part has to be developed
        
        #Within class loss and Between class loss
        additional_loss_func = self.calculate_additional_loss
        
        
        def ElopeLoss(y_true, y_pred):
            landa = self.loss_parameters['landa']
            
            # Within Class loss has to be computed first, in order to get the new class means updated
            
            additional_loss = additional_loss_func(y_true, y_pred)
            
            cat_cross_loss = categorical_crossentropy(y_true, y_pred)
            
            model_loss = tf.math.add(cat_cross_loss, tf.math.multiply(landa, additional_loss))
            
            return model_loss
        
        
        return ElopeLoss
    
    
    def compile(self):
        """ Compiles the Keras model. Includes metrics to differentiate between the two main loss terms """
        from tensorflow.keras.metrics import AUC
        self.model.compile(optimizer = self.optimizer, loss = self.loss_func,
                          metrics = [metrics.categorical_accuracy, self.calculate_additional_loss, categorical_crossentropy, AUC()])
        print('Model Compiled!')
        
    def save_model(self, name):
        """ Saves the model as a Json file"""
        # serialize model to JSON
        model_json = self.model.to_json()
        with open( str(name) + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(name + ".h5")
        print("Saved model to disk")    

    def load_trained_weights(self, name):
        """ Loads weights of a pre-trained model. 'weights' is path to h5 model\weights file"""
        json_file = open(str(name) + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, custom_objects = {'GlobalKMaxPooling2D': GlobalKMaxPooling2D})
        # load weights into new model
        loaded_model.load_weights( str(name) + ".h5")
        print("Loaded model from disk")      


# In[11]:


Elope = ElopeModel()


# In[ ]:

train_generator, validation_generator = load_generators_augmented()
history = Elope.model.fit_generator(train_generator, epochs = 35,
                   validation_data = validation_generator, callbacks = [lr_schedule, checkpoint],
                   verbose = 2)

l_param = Elope.loss_parameters['means']

name = 'Elope_xception_V3_flow'

# print(l_param.numpy())

Elope.save_model(name)

# Elope.load_trained_weights('Elope_modified_loss')
# In[ ]:


history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']

cat_acc = history_dict['categorical_accuracy']
val_cat_acc = history_dict['val_categorical_accuracy']

additional_loss = history_dict['calculate_additional_loss']
val_additional_loss = history_dict['val_calculate_additional_loss']

categorical_crossentropy = history_dict['categorical_crossentropy']
val_categorical_crossentropy = history_dict['val_categorical_crossentropy']



##############################
#  Saving information  #######
##############################

plt.ylim(0,5)
plt.plot(loss, 'r', label='Training loss')
plt.plot(val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig(name+"_loss.png")

plt.clf()

plt.ylim(0,1)
plt.plot(cat_acc, 'r', label='Training Categorical accuracy')
plt.plot(val_cat_acc, 'b', label='Validation Categorical accuracy')
plt.title('Training and validation Categorical accuracy')
plt.xlabel('epochs')
plt.ylabel('Categorical accuracy')
plt.legend()
plt.savefig(name+"_Categorical_accuracy.png")

plt.clf()

plt.ylim(0,5)
plt.plot(additional_loss, 'r', label='Training')
plt.plot(val_additional_loss, 'b', label='Validation')
plt.title('Training and Validation Within class loss')
plt.xlabel('epochs')
plt.ylabel('Within class loss')
plt.legend()
plt.savefig(name+"_within_class_loss.png")

plt.clf()

plt.ylim(0,2)
plt.plot(categorical_crossentropy, 'r', label='Training')
plt.plot(val_categorical_crossentropy, 'b', label='Validation')
plt.title('Training and Validation categorical_crossentropy')
plt.xlabel('epochs')
plt.ylabel('categorical_crossentropy')
plt.legend()
plt.savefig(name+"_categorical_crossentropy.png")







