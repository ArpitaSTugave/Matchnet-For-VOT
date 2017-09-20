# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:28:50 2017

@author: arpita
"""

#import matplotlib.pyplot as plt
import os.path
import leveldb, numpy as np, skimage
import cv2
import random
import itertools 
import settings
#import multiprocessing


 
def GetPatchFromGT(patch_ground_truth, container_dir,patch_height=64, patch_width=64):
    """Read patches from the given images.
    Returns
    -------
    image patch of 64x64.
    """
    
    if os.path.isfile(container_dir):
        cached_container_img = skimage.img_as_ubyte(skimage.io.imread( '%s'\
                                            %(container_dir), as_grey=True))
      
      
    X1,Y1,X2,Y2,X3,Y3,X4,Y4 = patch_ground_truth
    L=[[X1,Y1],[X2,Y2],[X3,Y3],[X4,Y4]]
    cnt = np.array(L).reshape((-1,1,2)).astype(np.int32)
    x,y,w,h = cv2.boundingRect(cnt)
    patch_image = cached_container_img[y:y+h, x:x+w]
    patch_image = cv2.resize(patch_image,(patch_height, patch_width), \
                    interpolation = cv2.INTER_LINEAR)
    patch_image = np.array(patch_image.reshape(1,patch_height,patch_width), \
                    dtype = np.float64)
    image_height, image_width = cached_container_img.shape

    
    #plt.imshow(patch_image,cmap=plt.get_cmap('gray'))
    #plt.show()
        
    # Extract the patch from the image and return.
    return  patch_image,w,h,image_height, image_width
    
    
def GetPatchUsingCenter(pair, h, w, container_dir,\
        patch_height=64, patch_width=64):
    """Read patches from the given images.
    Returns
    -------
    image patch of 64x64.
    """
    if os.path.isfile(container_dir):
        cached_container_img = skimage.img_as_ubyte(skimage.io.imread( '%s'\
                                            %(container_dir), as_grey=True))
    patch_image = cached_container_img[pair[0]-h/2:pair[0]+h/2, pair[1]-w/2:pair[1]+w/2];
    patch_image = cv2.resize(patch_image,(patch_height, patch_width), \
                    interpolation = cv2.INTER_LINEAR)
    patch_image = np.array(patch_image.reshape(1,patch_height,patch_width), \
                    dtype = np.float64)
    return patch_image
    
    
def GetRandomPatches(container_dir,real_width, real_height, \
    image_height, image_width, patch_height=64, patch_width=64, num_patches=1023):
    """
    Returns
    -------
    One 1023 * 1 * W * H array of random pacthes in current frame
    """ 
    #pool = multiprocessing.Pool()
    list1 = random.sample(range(real_height+1, image_height-real_height-1), patch_width/2)
    list2 = random.sample(range(real_width+1, image_width-real_width-1), patch_width/2)
    pairs = list(itertools.product(list1, list2))
    settings.pairs = pairs[:-1] # now the count is 1023
    return [GetPatchUsingCenter(x,  real_height, real_width,container_dir) for x in settings.pairs]
    #return [pool.map(GetPatchUsingCenter, pairs)]
    
    
    
    
def GetImagePatchesPreviousFrame(patch_ground_truth, container_dir, \
    num_patches=1024,patch_height=64, patch_width=64):
    """
    Returns
    -------
    One 1 * 1 * W * H array in a list of previous image frame
    """
    patch_image,_,_,_,_ = GetPatchFromGT(patch_ground_truth, container_dir)
    image_patchesl = [patch_image] * num_patches
    image_patches = np.asarray(image_patchesl)
    return image_patches
    
    
def GetImagePatchesCurrentFrame(patch_ground_truth, container_dir):
    """
    Returns
    -------
    One 1 * 1 * W * H array in a list of current image frame
    """
    
    # one of 1024 patches (target patch)
    patch_image_first,exact_patch_width,exact_patch_height,image_height, image_width = \
        GetPatchFromGT(patch_ground_truth, container_dir)
    patches_image = GetRandomPatches(container_dir, exact_patch_width, \
                        exact_patch_height,image_height, image_width)
    patches_image.insert(0, patch_image_first)
    image_patches = np.asarray(patches_image)
    return image_patches
    
    
def ReadPatches(ground_truth,i, image_previous, image_current, first_frame_only, \
    patch_height=64, patch_width=64):
    """Read patches from the given images.
    Returns
    -------
    Two N * 1 * W * H array in a list, where N is for current[1] and previous frame[0].
    """
    if first_frame_only :
        patches = [GetImagePatchesPreviousFrame(ground_truth[0],image_previous), \
                GetImagePatchesCurrentFrame(ground_truth[i+1],image_current)]
    else:
        patches = [GetImagePatchesPreviousFrame(ground_truth[i],image_previous), \
                GetImagePatchesCurrentFrame(ground_truth[i+1],image_current)]
                
    return patches
    
                        
    