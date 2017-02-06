# -*- coding: UTF-8 -*-
# from patchmatch import patch_match
import numpy as np
from PIL import Image
import skimage
from skimage import io
from skimage.morphology import skeletonize
from skimage import img_as_bool
from sklearn.neighbors import NearestNeighbors

from copy import copy, deepcopy
#import matplotlib.pyplot as plt
import time

import pandas as pd

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


S_contour_set = np.argwhere(S_contour==True)

T_contour = deepcopy(T)
for i in range(h):
    for j in range(w):
        if T[i,j] and np.count_nonzero(T[i-1:i+2,j-1:j+2]) >= 8:
            T_contour[i,j]=False


T_contour_set = np.argwhere(T_contour==True)

S_skel_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(S_skel_set)
distances, indices = S_skel_nbrs.kneighbors(S_contour_set)
y=np.sort(distances,axis=None)
x=range(len(distances))
#plt.scatter(x,y)
#plt.show()
k,b = np.polyfit(x, y, 1)
r_bar = 0.5*len(S_contour_set)*k+b

# Define Pixel class
class Pixel:
    pixelCount = 0
    def __init__(self, coordinate, distance, patchsize, patch, effect):
        self.coordinate = coordinate
        self.distance = distance
        self.patchsize = patchsize
        self.patch = patch
        self.effect = effect
        Pixel.pixelCount += 1

# Calculate patchsize for each pixel
h = S_prime.shape[0]
w = S_prime.shape[1]
S_pixels = []

patch_sizes=[29,25,21,17,13,9,5];
S_contour_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(S_contour_set)

t0 = time.clock()
for i in range(h-patch_sizes[0]):
    for j in range(w-patch_sizes[0]):
        patchsize = patch_sizes[0]
        max_effect=S_prime[i:i+patch_sizes[0],j:j+patch_sizes[0]]
        max_patch=S[i:i+patch_sizes[0],j:j+patch_sizes[0]]
        for s in patch_sizes:
            effect = S_prime[i:i+s,j:j+s]
            patch = S[i:i+s,j:j+s]
            if np.var(effect) > 0.02:
                patchsize = s
                max_effect = effect
                max_patch = patch
        d, index = S_contour_nbrs.kneighbors(np.array([i,j]).reshape(1,-1))
        if ~S[i,j]:
            distance = 1 + d / r_bar
        else:
            d_skel, ind = S_skel_nbrs.kneighbors(S_contour_set[index].reshape(1,-1))
            r_wave = max(d_skel, 0.2*len(S_contour_set)*k+b)
            distance = 1 - d / r_wave
            if distance < 0:
                distance = 0
        S_pixels.append(Pixel([i,j],distance,patchsize,max_patch,max_effect))


print time.clock() - t0, "seconds to calculate S_pixels"

# Probability table
distance_list = [float(a.distance) for a in S_pixels]
bins = range(101) * np.array(max(distance_list)) / 100
omega = np.digitize(distance_list,bins)
patchsize_list = [int(a.patchsize) for a in S_pixels]
table = df.groupby('bin')['patchsize'].value_counts() / df.groupby('bin')['patchsize'].count()



T_skel_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(T_skel_set)
distances, indices = T_skel_nbrs.kneighbors(T_contour_set)
y=np.sort(distances,axis=None)
x=range(len(distances))
#plt.scatter(x,y)
#plt.show()
k,b = np.polyfit(x, y, 1)
r_bar = 0.5*len(T_contour_set)*k+b

t0 = time.clock()
T_contour_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(T_contour_set)
T_pixels = []
for i in range(h-patch_sizes[0]):
    for j in range(w-patch_sizes[0]):
        d, index = T_contour_nbrs.kneighbors(np.array([i,j]).reshape(1,-1))
        if ~T[i,j]:
            distance = 1 + d / r_bar
        else:
            d_skel, ind = T_skel_nbrs.kneighbors(T_contour_set[index].reshape(1,-1))
            r_wave = max(d_skel, 0.2*len(T_contour_set)*k+b)
            distance = 1 - d / r_wave
            if distance < 0:
                distance = 0
        T_pixels.append(Pixel([i,j],distance,patchsize,max_patch,max_effect))


print time.clock() - t0, "seconds to calculate T_pixels"

t0 = time.clock()
T_prime = np.zeros([h, w , 3],dtype=np.float64) + 9999
for i in range(h-patch_sizes[0]):
    for j in range(w-patch_sizes[0]):
        if np.array_equal(T_prime[i,j],[9999,9999,9999]):
            print i,j
            patch_distance = 9999
            size = 0
            patch = []
            for p in S_pixels:
                if np.linalg.norm(p.patch-T[i:i+p.patchsize,j:j+p.patchsize]) < patch_distance:
                    patch_distance = np.linalg.norm(p.patch-T[i:i+p.patchsize,j:j+p.patchsize])
                    patch = p
            T_prime[i:i+patch.patchsize,j:j+patch.patchsize] = patch.effect

for i in range(h):
    for j in range(w):
        if np.array_equal(T_prime[i,j],[9999,9999,9999]):
            T_prime[i,j] = [0,0,0]

print time.clock() - t0, "seconds to calculate T_prime"
io.imsave("T_prime.jpg", T_prime)
io.imshow(T_prime)
io.show()
