from patchmatch import patch_match
import numpy as np
from PIL import Image
import skimage
from skimage import io
from skimage.morphology import skeletonize
from skimage import img_as_bool

# Read S and T as binary images
S = img_as_bool(io.imread("S.jpg", as_grey=True))
io.imshow(S)
io.show()
T = img_as_bool(io.imread("T.jpg", as_grey=True))
io.imshow(T)
io.show()

# Read S_prime as float image
S_prime = skimage.img_as_float(io.imread("S_prime.jpg"))
io.imshow(S_prime)
io.show()

# Get skeleton of S & T
S_skel = skeletonize(S)
io.imshow(S_skel)
io.show()

T_skel = skeletonize(T)
io.imshow(T_skel)
io.show()

P=T_skel[100,100]
