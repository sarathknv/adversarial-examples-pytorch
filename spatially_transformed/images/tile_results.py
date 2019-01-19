import cv2
import os
import numpy as np

images = os.listdir('results/')
images.sort()

arr = np.zeros((28*10, 28*10), dtype=np.uint8)

for i, image in enumerate(images):
    img_path = os.path.join('results', image)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    a = int(image.split('_')[0])
    b = int(image.split('_')[1].split('.')[0])

    arr[a*28: (a+1)*28, b*28: (b+1)*28] = img


import pdb; pdb.set_trace()
