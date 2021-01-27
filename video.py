import os
import cv2
import numpy as np
from tqdm import tqdm

DATA = 'tracking-data'


def gen_video_from_images():
    files = sorted(os.listdir(DATA))
    img = cv2.imread(os.path.join(DATA, files[0]))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('input_video.avi',fourcc, 20.0, tuple([img.shape[1], img.shape[0]]))

    for i in tqdm(files):
        img = cv2.imread(os.path.join(DATA, i))
        out.write(img)

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()