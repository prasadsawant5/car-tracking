import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas as pd
from video import gen_video_from_images

DATA = 'tracking-data'
INPUT_VIDEO = 'input_video.avi'
OUTPUT_VIDEO = 'output.mp4'
OUTPUT_PD = 'output-points.csv'
HSVLOW = np.array([79, 0, 102])
HSVHIGH = np.array([164, 57, 253])

hsv = None
res = None



def obtain_contours(img):
    frame = cv2.GaussianBlur(img, (5, 5), 0)

    #convert from a BGR stream to an HSV stream
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #create a mask for that range
    mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)

    res = cv2.bitwise_and(frame, frame, mask =mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, 1, 2)

    area = []
    for cnt in contours:
        a = cv2.contourArea(cnt)
        area.append(a)

    max_index = area.index(max(area))
    cnt = contours[max_index]
    x, y, w, h = cv2.boundingRect(cnt)
    return [x, y, w, h]

if __name__ == '__main__':
    if not os.path.exists(INPUT_VIDEO):
        gen_video_from_images()

    is_first_frame = True
    tracker = cv2.TrackerKCF_create()
    cap = cv2.VideoCapture(INPUT_VIDEO)

    # Define the codec and create VideoWriter object for output video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')


    datapoints = []

    roi = None
    frame_counter = 1
    while cap.isOpened():
        ret, img = cap.read()

        if ret:
            if is_first_frame:
                is_first_frame = False
                x, y, w, h = obtain_contours(img)
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                roi = img[x:x+w, y:y+h, :]

                tracker.init(img, (x, y, w, h))
                out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, tuple([img.shape[1], img.shape[0]]))

                datapoints.append([frame_counter, x, y])
            else:
                status, bbox = tracker.update(img)
                (x, y, w, h) = [int(v) for v in bbox]
                
                if status:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
                    datapoints.append([frame_counter, x, y])

            out.write(img)

            frame_counter += 1
        else:
            break

    df = pd.DataFrame(datapoints, columns=['Frame No', 'x', 'y'])
    df.to_csv(OUTPUT_PD, index=False)

    cap.release()
    out.release()

