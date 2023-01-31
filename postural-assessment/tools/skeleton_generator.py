import cv2
import mediapipe as mp
import numpy as np
import json
import os
import matplotlib.pyplot as plt

mmp = [0, -1, -1, -1, -1, -1, -1, 1, 2, -1, -1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22,
       23, 24, 25, 26]
con = [[0, 1], [0, 2], [0, 3], [3, 4], [3, 5], [3, 16], [4, 6], [5, 7], [6, 8], [7, 9], [8, 14], [9, 15], [8, 12],
       [9, 13], [8, 10], [9, 11], [10, 12], [11, 13], [16, 17], [16, 18], [17, 19], [18, 20], [19, 21],
       [20, 22], [21, 23], [22, 24], [23, 25], [24, 26], [21, 25], [22, 26]]


def change_coordinate(skeleton):
    mid_node = (skeleton[4] + skeleton[5] + skeleton[17] + skeleton[18]) / 4
    for i, node in enumerate(skeleton):
        skeleton[i][:3] -= mid_node[:3]
    return skeleton


def draw_skeleton(img, skeleton, middle=True, video=False):
    if middle:
        skeleton = change_coordinate(skeleton)
    h, w, c = img.shape

    # draw bones
    for i in con:
        bx, by = int(skeleton[i[0]][0] * w), int(skeleton[i[0]][1] * h)
        ex, ey = int(skeleton[i[1]][0] * w), int(skeleton[i[1]][1] * h)
        if middle:
            bx, by = int(bx + w / 2), int(by + h / 2)
            ex, ey = int(ex + w / 2), int(ey + h / 2)
        # if skeleton[i[0]][3] > 0.2 and skeleton[i[1]][3] > 0.2:
        cv2.line(img, (bx, by), (ex, ey), (0, 255, 0), int(h / 100))

    # draw joints
    for id, lm in enumerate(skeleton):
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        if middle:
            cx, cy = int(cx + w / 2), int(cy + h / 2)
        cv2.circle(img, (cx, cy), int(h / 100), (0, 0, 255), cv2.FILLED)
    cv2.imshow('', img)
    cv2.waitKey(1 if video else 0)


def read_skeleton_video(path):
    skeleton_video = list()
    with open(path, 'r') as load_f:
        load_list = json.load(load_f)
        print('load {} success'.format(path))
    for dict in load_list:
        skeleton = dict['skeleton']
        skeleton_video.append(skeleton)
    return skeleton_video


def make_skeleton_video(path):
    mpPose = mp.solutions.pose
    skeleton_video = list()
    cap = cv2.VideoCapture(path)
    frame = 0
    while True:
        success, img = cap.read()
        if not success:
            break
        skeleton = get_skeleton(mpPose, img)
        skeleton_video.append(skeleton)
        frame += 1
        print('frame {} success!'.format(frame))
    return skeleton_video


def get_skeleton(mpPose, img):
    pose = mpPose.Pose()
    # img = cv2.resize(img, (1000, 1600))

    h, w, c = img.shape
    # print(img.shape)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # get original skeleton
    result = pose.process(imgRGB).pose_landmarks
    skeleton = np.zeros((27, 4))
    if result:
        # build new skeleton
        for id, lm in enumerate(result.landmark):
            if mmp[id] != -1:
                skeleton[mmp[id]] = [lm.x, lm.y, lm.z, lm.visibility]
        mid = (skeleton[4] + skeleton[5] + skeleton[17] + skeleton[18]) / 4
        skeleton[3] = (mid + skeleton[4] + skeleton[5]) / 3
        skeleton[16] = (mid + skeleton[17] + skeleton[18]) / 3
    return skeleton


def skeleton2json(skeleton, dir):
    skeleton_list = skeleton.tolist()

    output_dict = dict()
    output_dict['skeleton'] = skeleton_list

    if not os.path.exists(dir):
        os.mkdir(dir)
    path = '{}skeleton.json'.format(dir)
    with open(path, "w") as f:
        json.dump(output_dict, f)
        print('write {} success'.format(path))

def skeleton_video2json(skeleton_video, dir):
    output_list = list()
    if not os.path.exists(dir):
        os.mkdir(dir)
    path = '{}skeleton_video.json'.format(dir)

    for frame, skeleton in enumerate(skeleton_video):
        skeleton_list = skeleton.tolist()

        output_dict = dict()
        output_dict['frame'] = frame
        output_dict['skeleton'] = skeleton_list

        output_list.append(output_dict)

    with open(path, "w") as f:
        json.dump(output_list, f)
        print('write {} success'.format(path))


def test():
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    img_path = './test_images/test7.jpg'
    img_output_dir = './test_images/output/'
    video_path = './test_videos/test1.mov'
    video_output_dir = './test_videos/output/'

    video_json_path = './test_videos/output/skeleton_video.json'

    # l = make_skeleton_video(video_path)
    # skeleton_video2json(l, video_output_dir)
    l = read_skeleton_video(video_json_path)
    img = np.ndarray((1080, 1920, 3))
    for sk in l:
        draw_skeleton(img, sk, middle=True, video=True)
    print(l)
    # img = cv2.imread(path)
    # sk = get_skeleton(mpPose, img)
    #
    # skeleton2json(sk, output_dir)
    # draw_skeleton(img, sk, middle=True)

    # print(sk)
