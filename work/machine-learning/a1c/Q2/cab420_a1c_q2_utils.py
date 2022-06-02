#
# Utility functions for CAB420, Assignment 1C, Q2
# Author: Simon Denman (s.denman@qut.edu.au)
#

import pandas
import cv2
import os
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt     # for plotting

# load a set of masks that represent the semantic regions of a person. This will load the image into a 
# (width, height, 8) array if merge_skin is false; or a (width, height, 6) array is merge_skin is true
#   base_path:  path to the images
#   base_name:  common name for the images
#   merge_skin: bool to indicate if we are going to merge the skin regions into a single channel, or
#               leave them as arms, face, legs
#
#   returns:    a (width, height, 9) or (width, height, 7) image. Really it's just all the binary masks
#               stacked up
#
def load_mask(base_path, base_name, merge_skin = False):
    
    # hard-coding warning, but really dataset structure can be considered a bit of a constant
    components = ['_hair', '_legs', '_luggage', '_shoes', '_torso', '_skin_arms', '_skin_face', '_skin_legs']
    # loop through and load all the components
    images = []
    for c in components:
        images.append(cv2.imread(os.path.join(base_path, base_name + c + '.png'), cv2.IMREAD_GRAYSCALE) / 255.0)            
       
    # this is a bit ugly, but we know that we have skin in the last three channels, so we can just add
    # those together and then cut the last two channels out
    if (merge_skin):
        images[-3] = images[-3] + images[-2] + images[-1]
        images = images[:-2]        
    
    # convert our list to a numpy array, and then transpose it to get the channels in the correct spot
    return numpy.transpose(numpy.array(images), (1, 2, 0))

# adds a background channel to the mask. This channel is 1.0 (i.e on) when all other channels are 0.0 (off), and
# indicates that there's nothing there. This may seem odd, but is needed if we want to use this mask for a
# semantic segmentation output, in which case we'd need a channel to indicate the absence of anything interesting, 
# i.e. the backgorund.
#  mask:    mask to add a background channel to
#  
#  returns: mask image with a new channel
#
def add_background_channel_to_mask(mask):
    # create the background
    background = 1.0 - mask[:,:,0]
    for i in range(1, mask.shape[2]):
        background = background - mask[:,:,i]

    # stack the background with the original mask. You can put the background first or last
    # (or in the middle - but first or last makes more sense I think), it doesn't really matter
    return numpy.dstack((mask, background))

# load a dataset. This will optionally load masks, and resize images (and masks) and convert images to a new
# colour space
#   csv_path:      path to the csv file that contains all the meta-data, ground truth, etc
#   image_path:    path to the image of the people for whom we need to recognise traits
#   mask_path:     where are the mask images? If there are none, set this to None
#   target_size:   what size images do you want? This will resize and pad, so will preserve the aspect ratio.
#   target_colour: what colour space do you want to convert to? Default is cv2.COLOR_BGR2RGB to go from
#                  opencv's default backwards world to RGB, but you may also want cv2.COLOR_BGR2GRAY 
#                  (though, don't you need to recognise colour?), or something more exotic like HSV or LAB.1
#   merge_skin:    bool to indicate if we are going to merge the skin regions into a single channel, or
#                  leave them as arms, face, legs. Only does anything if mark_path is not None
#
#   returns        two dictionaries: x, which contains the images; and y, which contains all the target things
#                  (gender, colours, etc) and potentially the masks
#
def load_set(csv_path, image_path, mask_path = None, target_size = (60, 100), target_colour = cv2.COLOR_BGR2RGB, merge_skin = False):
    
    # create storage
    # first for likley model inputs
    x = {}
    x['images'] = []

    # now for likely model outputs
    y = {}
    y['gender'] = []
    y['torso_type'] = []
    y['torso_colour'] = []
    y['leg_type'] = []
    y['leg_colour'] = []
    y['luggage'] = []
    if (mask_path is not None):
        y['mask'] = []

    # load the csv file, which contains paths to images and all the ground truth data
    csv = pandas.read_csv(csv_path)
    for i, row in csv.iterrows():
        # there's a bit going on here, working from the inside out:
        #  - load the file
        #  - convert the colour space
        #  - resize with padding (using tensorflow)
        #  - convert back to numpy (because we used tensorflow)
        x['images'].append(tf.image.resize_with_pad(cv2.cvtColor(cv2.imread(os.path.join(image_path, row['filename'])), target_colour) / 255.0, target_size[1], target_size[0]).numpy())

        # pull out the various bits of ground truth we need
        y['gender'].append(row['gender'])
        y['torso_type'].append(row['tortyp'])
        y['torso_colour'].append(row['torcol'])
        y['leg_type'].append(row['legtyp'])
        y['leg_colour'].append(row['legcol'])
        y['luggage'].append(row['luggage'])
        # is there some other piece of data you want to use? maybe the pose? or some other label? if so then just pull it out too!        
        
        # pull out the mask, if we are doing that
        if (mask_path is not None):
            # a bit like the above, we'll
            #  - load the image
            #  - resize and pad the image
            #  - convert back to numpy
            #  - add a background channel
            # The background channel is done last to ensure that the newly added padded regions are included as 
            # background (i.e. contain nothing of interest)
            y['mask'].append(add_background_channel_to_mask(tf.image.resize_with_pad(load_mask(mask_path, os.path.splitext(row['filename'])[0], merge_skin), target_size[1], target_size[0]).numpy()))

    x['images'] = numpy.array(x['images'])
    for key in y:
        y[key] = numpy.array(y[key])
    
    return x, y

# load the data
#   base_path:     the path to the data, within the directory that this points to there should be the Train_Data 
#                  and Test_Data directories
#   target_size:   what size images do you want? This will resize and pad, so will preserve the aspect ratio.
#   target_colour: what colour space do you want to convert to? Default is cv2.COLOR_BGR2RGB to go from
#                  opencv's default backwards world to RGB, but you may also want cv2.COLOR_BGR2GRAY 
#                  (though, don't you need to recognise colour?), or something more exotic like HSV or LAB.1
#   merge_skin:    bool to indicate if we are going to merge the skin regions into a single channel, or
#                  leave them as arms, face, legs. Only does anything if mark_path is not None
#
#   returns:       loaded training and testing data
#
def load_data(base_path, target_size = (60, 100), target_colour = cv2.COLOR_BGR2RGB, merge_skin = False):
    
    train_x, train_y = load_set(os.path.join(base_path, 'Train_Data', 'Train.csv'), os.path.join(base_path, 'Train_Data', 'Originals'), os.path.join(base_path, 'Train_Data', 'Binary_Maps'), target_size = target_size, target_colour = target_colour, merge_skin = merge_skin)
    test_x, test_y = load_set(os.path.join(base_path, 'Test_Data', 'Test.csv'), os.path.join(base_path, 'Test_Data', 'Originals'), target_size = target_size, target_colour = target_colour, merge_skin = merge_skin)    
    
    return train_x, train_y, test_x, test_y

# does what it says on the box, makes a mask a bit prettier and a lot more practical for display
#   mask_image: input, N channel mask image
#
#   returns:    mask collapsed to a single channel, channels have been weighted and summed such that each
#               channel should occupy it's own space in the colour map of your choice
#
def make_a_mask_image_ready_for_display(mask_image):
    num_channels = mask_image.shape[2]
    im = mask_image[:,:,0]*(1.0/mask_image.shape[2])
    for i in range(1, num_channels):
        im = im + mask_image[:,:,i]*(i/mask_image.shape[2])
    return im

# Plot some images. Will plot the first 50 samples in a 10x5 grid
#  x: array of images, of shape (samples, width, height, channels)
#
def plot_images(images):
    fig = plt.figure(figsize=[15, 18])
    for i in range(50):
        ax = fig.add_subplot(5, 10, i + 1)
        ax.imshow(images[i,:], cmap=plt.get_cmap('Greys'))
        ax.axis('off') 

# Plot some images and their masks. Will plot the first 25 samples in a 10x5 grid,
# alternating between images and masks
#  x: array of images, of shape (samples, width, height, channels)
#  masks: array of masks, of shape (samples, width, height, channels)
#
def plot_images_and_masks(images, masks):
    fig = plt.figure(figsize=[15, 18])
    for i in range(25):
        ax = fig.add_subplot(5, 10, i*2 + 1)
        ax.imshow(images[i,:], cmap=plt.get_cmap('Greys'))
        ax.axis('off') 
        ax = fig.add_subplot(5, 10, i*2 + 2)
        ax.imshow(make_a_mask_image_ready_for_display(masks[i,:]), cmap=plt.get_cmap('RdYlBu'))
        ax.axis('off')         
        