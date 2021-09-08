#!/usr/bin/env python
# coding: utf-8

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.layers import *
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K

import numpy as np
import os
import random
import scipy.misc
from tqdm import *
from random import sample

def load_dataset_small(train_dir, test_dir, img_shape):
    """Loads training and test datasets, from Tiny ImageNet Visual Recogition Challenge.

    Arguments:
        num_images_per_class_train: number of images per class to load into training dataset.
        num_images_test: total number of images to load into training dataset.
    """
    X_train = []
    X_test = []
    
    # Create training set.
    for sub in os.listdir(train_dir):
        sub_dir = os.path.join(train_dir, sub)
        c_imgs = os.listdir(sub_dir)
        for img_name_i in c_imgs:
            try:
                img_i = image.load_img(os.path.join(sub_dir, img_name_i), target_size=img_shape)
                x = image.img_to_array(img_i)
                X_train.append(x)
            except Exception as e:
                print(str(e))
                continue
    
    # Create test set.
    test_dir = os.path.join(test_dir)
    test_imgs = os.listdir(test_dir)
    for img_name_i in test_imgs:
        img_i = image.load_img(os.path.join(test_dir, img_name_i), target_size=img_shape)
        x = image.img_to_array(img_i)
        X_test.append(x)
    
    # Return train and test data as numpy arrays.
    return np.array(X_train), np.array(X_test)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# Loss for reveal network
def rev_loss(s_true, s_pred):
    beta = 1.0
    # Loss for reveal network is: beta * |S-S'|
    return beta * root_mean_squared_error(s_true, s_pred)

# Loss for the full model, used for preparation and hidding networks
def full_loss(y_true, y_pred):
    # Loss for the full model is: |C-C'| + beta * |S-S'|
    s_true, c_true = y_true[...,0:3], y_true[...,3:6]
    s_pred, c_pred = y_pred[...,0:3], y_pred[...,3:6]
    
    s_loss = rev_loss(s_true, s_pred)
    c_loss = root_mean_squared_error(c_true, c_pred)
    
    return s_loss + c_loss


# Returns the encoder as a Keras model, composed by Preparation and Hiding Networks.
def make_encoder(input_size):
    input_S = Input(shape=(input_size))
    input_C= Input(shape=(input_size))

    # Preparation Network
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_3x3')(input_S)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_4x4')(input_S)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_prep0_5x5')(input_S)
    #x = concatenate([input_C, x3, x4, x5])
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_prep1_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    x = concatenate([input_C, x])
    
    # Hiding network
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid0_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid0_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid0_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid1_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid1_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid1_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_hid2_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_hid2_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_hid2_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    output_C = Conv2D(3, (3, 3), strides = (1, 1), padding='same', activation='relu', name='output_C')(x)
    
    return Model(inputs=[input_S, input_C],
                 outputs=output_C,
                 name = 'Encoder')

# Returns the decoder as a Keras model, composed by the Reveal Network
def make_decoder(input_size):
    
    # Reveal network
    reveal_input = Input(shape=(input_size))
    
    # Adding Gaussian noise with 0.01 standard deviation.
    input_with_noise = GaussianNoise(0.01, name='output_C_noise')(reveal_input)
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_3x3')(input_with_noise)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_4x4')(input_with_noise)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev0_5x5')(input_with_noise)
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev1_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    x3 = Conv2D(50, (3, 3), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_3x3')(x)
    x4 = Conv2D(10, (4, 4), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_4x4')(x)
    x5 = Conv2D(5, (5, 5), strides = (1, 1), padding='same', activation='relu', name='conv_rev2_5x5')(x)
    x = concatenate([x3, x4, x5])
    
    output_S = Conv2D(3, (3, 3), strides = (1, 1), padding='same', activation='relu', name='output_S')(x)
    
    return Model(inputs=reveal_input, outputs=output_S, name = 'Decoder')

def make_model(input_size):
    input_S = Input(shape=(input_size))
    input_C= Input(shape=(input_size))
    
    encoder = make_encoder(input_size)
    
    decoder = make_decoder(input_size)
    decoder.compile(optimizer='adam', loss=rev_loss)
    decoder.trainable = False
    
    output_C = encoder([input_S, input_C])
    output_S = decoder(output_C)

    full_model = Model(inputs=[input_S, input_C],
                        outputs=concatenate([output_S, output_C]))
    full_model.compile(optimizer='adam', loss=full_loss)
    
    return encoder, decoder, full_model

def lr_schedule(epoch_idx):
    if epoch_idx < 200:
        return 0.001
    elif epoch_idx < 400:
        return 0.0003
    elif epoch_idx < 600:
        return 0.0001
    else:
        return 0.00003


if __name__ == "__main__":

    ### Constants ###
    data_dir = "."
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    img_shape = (256, 256)

    # Load dataset.
    X_train_orig, X_test_orig = load_dataset_small(train_dir, test_dir, img_shape)
    
    # Normalize image vectors.
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    
    train_index = sample([i for i in range(len(X_train))], int(len(X_train)*0.8))
    valid_index = list(set(i for i in range(len(X_train))) - set(train_index))
    X_valid = X_train[valid_index]
    X_train = X_train[train_index]
    
    # Print statistics.
    print ("Number of training examples = " + str(X_train.shape[0]))
    print ("Number of validation examples = " + str(X_valid.shape[0]))
    print ("Number of test examples = " + str(X_train.shape[0]))
    print ("X_train shape: " + str(X_train.shape)) # Should be (train_size, 64, 64, 3).
    print ("X_valid shape: " + str(X_valid.shape)) # Should be (train_size, 64, 64, 3).

    
    
    # S: secret image
    input_S = X_train[0:X_train.shape[0] // 2]
    
    # C: cover image
    input_C = X_train[X_train.shape[0] // 2:]
    
    valid_S = X_valid[0:X_valid.shape[0] // 2]
    valid_C = X_valid[X_valid.shape[0] // 2:]

    encoder, decoder, full_model = make_model(input_S.shape[1:])


    description = 'Epoch {} | Batch: {:3} of {}. Training loss FM {:10.2f} DE {:10.2f} | Validation loss FM {:10.2f} EN {:10.2f} DE {:10.2f}'
    
    epoches = 20
    batch_size = 32
    
    m = input_S.shape[0]
    train_fm_loss_history = []
    train_de_loss_history = []
    valid_fm_loss_history = []
    valid_en_loss_history = []
    valid_de_loss_history = []
    
    
    for epoch in range(epoches):
        np.random.shuffle(input_S)
        np.random.shuffle(input_C)
        
        t = tqdm(range(0, input_S.shape[0], batch_size), mininterval=0)
        train_fm_loss_l = []
        train_de_loss_l = []
        
        valid_fm_loss_l = []
        valid_en_loss_l = []
        valid_de_loss_l = []
        for idx in t:
            
            batch_S = input_S[idx:min(idx + batch_size, m)]
            batch_C = input_C[idx:min(idx + batch_size, m)]
            
            C_prime = encoder.predict([batch_S, batch_C])
            
            train_fm_loss_l.append(full_model.train_on_batch(x=[batch_S, batch_C],
                                                       y=np.concatenate((batch_S, batch_C),axis=3)))
            train_de_loss_l.append(decoder.train_on_batch(x=C_prime,
                                                  y=batch_S))
            
            decoded = full_model.predict([valid_S, valid_C])
            decoded_S, decoded_C = decoded[...,0:3], decoded[...,3:6]
            
            
            valid_fm_loss_l.append(full_loss(np.concatenate((valid_S, valid_C),axis=3), decoded))
            valid_en_loss_l.append(root_mean_squared_error(decoded_C, valid_C))
            valid_de_loss_l.append(rev_loss(valid_S, decoded_S))
    
            # Update learning rate
            K.set_value(full_model.optimizer.lr, lr_schedule(epoch))
            K.set_value(decoder.optimizer.lr, lr_schedule(epoch))
            
            t.set_description(description.format(epoch + 1, idx, m, np.mean(train_fm_loss_l), np.mean(train_de_loss_l), np.mean(valid_fm_loss_l), np.mean(valid_en_loss_l), np.mean(valid_de_loss_l)))
        train_fm_loss_history.append(np.mean(train_fm_loss_l))
        train_de_loss_history.append(np.mean(train_de_loss_l))
        valid_fm_loss_history.append(np.mean(valid_fm_loss_l))
        valid_en_loss_history.append(np.mean(valid_en_loss_l))
        valid_de_loss_history.append(np.mean(valid_de_loss_l))
    
    
    np.savetxt("train_fm_loss_history.txt", train_fm_loss_history, delimiter=",")
    np.savetxt("train_de_loss_history.txt", train_de_loss_history, delimiter=",")
    np.savetxt("valid_fm_loss_history.txt", valid_fm_loss_history, delimiter=",")
    np.savetxt("valid_en_loss_history.txt", valid_en_loss_history, delimiter=",")
    np.savetxt("valid_de_loss_history.txt", valid_de_loss_history, delimiter=",")

    # Save model
    full_model.save_weights('models/weights.hdf5')

