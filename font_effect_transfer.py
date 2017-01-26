# -*- coding: UTF-8 -*-
from patchmatch import patch_match
import numpy as np
from PIL import Image
import skimage
from skimage import io
from skimage.morphology import skeletonize
from skimage import img_as_bool
from sklearn.neighbors import NearestNeighbors

from copy import copy, deepcopy

# Height and width of image
h = S_prime.shape[0]
w = S_prime.shape[1]

# Read S and T as binary images
S = img_as_bool(io.imread("S.jpg", as_grey=True))
#io.imshow(S)
#io.show()
T = img_as_bool(io.imread("T.jpg", as_grey=True))
#io.imshow(T)
#io.show()

# Read S_prime as float image
S_prime = skimage.img_as_float(io.imread("S_prime.jpg"))
#io.imshow(S_prime)
#io.show()

# Get skeleton set of S & T
S_skel = skeletonize(S)
#io.imshow(S_skel)
#io.show()
S_skel_set = np.argwhere(S_skel==True)

T_skel = skeletonize(T)
#io.imshow(T_skel)
#io.show()
T_skel_set = np.argwhere(T_skel==True)


# Get contour set of S & T
S_contour = deepcopy(S)
for i in range(h):
    for j in range(w):
        if S[i,j] and np.count_nonzero(S[i-1:i+2,j-1:j+2]) >= 8:
            S_contour[i,j]=False

T_contour = deepcopy(T)
for i in range(h):
    for j in range(w):
        if T[i,j] and np.count_nonzero(T[i-1:i+2,j-1:j+2]) >= 8:
            T_contour[i,j]=False

# Define Pixel class
class Pixel:
    pixelCount = 0
    def __init__(self, coordinate, distance, patchsize, patch):
        self.coordinate = coordinate
        self.distance = distance
        self.patchsize = patchsize
        self.patch = patch
        Pixel.pixelCount += 1

# Calculate patchsize for each pixel
h = S_prime.shape[0]
w = S_prime.shape[1]
S_pixels = []

patch_sizes=[21,19,17,15,13,11,9,7,5];
for i in range(h-21):
    for j in range(w-21):
        patchsize = 21
        max_patch=S_prime[i:i+21,j:j+21]
        for s in patch_sizes:
            patch = S_prime[i:i+s,j:j+s]
            if np.var(patch) > 0.02:
                patchsize = s
                max_patch = patch
        S_pixels.append(Pixel([i,j],0,patchsize,max_patch))


#nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
