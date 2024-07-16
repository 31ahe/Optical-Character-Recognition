import config as cfg
import cv2 as cv
import pathlib
import random
import os

img_list = list(pathlib.Path(os.path.join(cfg.test_path, "Images")).glob('*.jpg'))

dst = "path/to/noisy_images"

for fname in img_list:
    img = cv.imread(str(fname), 0)
    for _ in range(3):
        pt1 = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
        pt2 = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
        noisy_img = cv.line(img, pt1, pt2, color=0, thickness=1)
        
    cv.imwrite(os.path.join(dst, f'{fname.stem}.jpg'), noisy_img)