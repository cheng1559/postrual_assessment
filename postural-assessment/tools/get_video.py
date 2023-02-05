import cv2
import os
import numpy as np
import time


def get_video(save_dir, length, delay=5000, show=False):
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    h, w, c = img.shape

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fps = 30
    save_path = '{}video.mp4'.format(save_dir)
    _size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, _size, True)

    t = time.perf_counter()

    last_t = 0
    while True:
        cur_t = int((time.perf_counter() - t) * 1000)
        success, img = cap.read()
        if not success:
            break

        if cur_t <= delay:
            if int(last_t / 1000) != int(cur_t / 1000):
                print('Video recording will start in {}s.'.format(int(delay / 1000) - int(cur_t / 1000)))
        elif cur_t <= delay + length:
            if int(last_t / 1000) != int(cur_t / 1000):
                print('Video recording has been in progress for {}s.'.format(int(cur_t / 1000) - int(delay / 1000)))
            videoWriter.write(img)
        else:
            print('Video has been saved as {}.'.format(save_path))
            videoWriter.release()
            break
        last_t = cur_t

        # cv2.imshow('', img)
        cv2.waitKey(1)
    cap.release()