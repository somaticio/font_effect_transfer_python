import numpy as np
from PIL import Image
import skimage
from skimage import io
from skimage.morphology import skeletonize
from skimage import img_as_bool
from sklearn.neighbors import NearestNeighbors

import random
import sys
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import time

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

# Height and width of image
h = S_prime.shape[0]
w = S_prime.shape[1]

def get_contour(font):
    contour = deepcopy(font)
    for i in range(h):
        for j in range(w):
            if font[i,j] and np.count_nonzero(font[i-1:i+2,j-1:j+2]) >= 8:
                contour[i,j]=False
    return contour

def patch(font, location, patchsize):
    row = location[0]
    col = location[1]
    r = patchsize/2
    return font[row-r-1:row+r, col-r-1:col+r]
# Get coordinate set of S & T
S_set = np.argwhere(S==True)
T_set = np.argwhere(T==True)

# Get contour coordinate set of S & T
S_contour = get_contour(S)
S_contour_set = np.argwhere(S_contour==True)
T_contour = get_contour(T)
T_contour_set = np.argwhere(T_contour==True)
T_contour_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(T_contour_set)
T_prime = np.zeros([h, w , 3],dtype=np.float64)
for t in T_set:
    if np.array_equal(T_prime[t[0],t[1]],[0,0,0]):
        T_patch = patch(T, t, 5)
        distance = sys.maxint
        source_location = []
        effect = []
        for i in range(100):
            location = random.choice(S_set)
            S_patch = patch(S, location, 5)
            if np.linalg.norm(S_patch-T_patch) < distance:
                distance = np.linalg.norm(S_patch-T_patch)
                target_location = location
            T_prime[t[0]-5/2-1:t[0]+5/2, t[1]-5/2-1:t[1]+5/2]=patch(S_prime,target_location,5)

io.imshow(T_prime)
io.show()


# TODO: Get effect on contour
# TODO: Get effect outside font
# TODO: Better overlaping patches
