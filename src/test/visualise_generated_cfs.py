import matplotlib.pyplot as plt
import random
import numpy as np
from utils.params import *

w = 30
h = 10
columns = 1
rows = 2
ax = []

originals = np.load("data/original_imgs.npy")
cfs = np.load("data/cf_imgs.npy")
print(len(originals))
for i in range(10):
    fig = plt.figure(figsize=(8, 8))
    idx = random.randint(0, len(originals)-1)
    original_img = originals[idx]
    cf_img = cfs[idx]
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(original_img)
    axarr[1].imshow(cf_img)
    
    plt.show()
