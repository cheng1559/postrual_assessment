import cv2
import mediapipe as mp
import numpy as np
import json
import os
import math
import tools.dtw_calculate as dc
import matplotlib.pyplot as plt

mmp = [0, -1, -1, -1, -1, -1, -1, 1, 2, -1, -1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22,
       23, 24, 25, 26]
con = [[0, 1], [0, 2], [0, 3], [3, 4], [3, 5], [3, 16], [4, 6], [5, 7], [6, 8], [7, 9], [8, 14], [9, 15], [8, 12],
       [9, 13], [8, 10], [9, 11], [10, 12], [11, 13], [16, 17], [16, 18], [17, 19], [18, 20], [19, 21],
       [20, 22], [21, 23], [22, 24], [23, 25], [24, 26], [21, 25], [22, 26]]


def change_coordinate(skeleton):
    mid_joint = (skeleton[4] + skeleton[5] + skeleton[17] + skeleton[18]) / 4
    for i, joint in enumerate(skeleton):
        skeleton[i][:3] -= mid_joint[:3]
    return skeleton


def change_video_coordinate(skeleton_video):
    sk = skeleton_video[0][0][0]
    frame = 0
    for i, skeleton in enumerate(skeleton_video):
        if skeleton[0][0] != 0:
            sk += skeleton
            frame += 1
    if frame != 0:
        sk /= frame
    else:
        print('From change_video_coordinate: no skeleton detected!')
        return []

    mid_joint = (sk[4] + sk[5] + sk[17] + sk[18] + sk[0] + sk[26] + sk[25] + sk[10] + sk[11]) / 9
    n1, n2 = sk[3], sk[16]
    dis = math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)
    p = 0.1 / dis if dis != 0 else 1

    for i, skeleton in enumerate(skeleton_video):
        n = min(10, len(skeleton_video) - i)
        for j in range(1, n):
            skeleton_video[i] += skeleton_video[i + j]
        skeleton_video[i] /= n
        # if i > 0 and sk_distance(skeleton_video[i], skeleton_video[i - 1]) > 40:
        #     skeleton_video[i] = skeleton_video[i - 1]

        for j, joint in enumerate(skeleton):
            skeleton_video[i][j][:3] -= mid_joint[:3]
            skeleton_video[i][j][:3] *= p

    return skeleton_video


def draw_skeleton(img, skeleton, middle=True, show=False, video=False):
    # if middle:
    #     skeleton = change_coordinate(skeleton)
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
    if show:
        cv2.imshow('', img)
        cv2.waitKey(1 if video else 0)
    return img


def read_skeleton_video(dir, json_name, debug=False):
    path = '{}{}'.format(dir, json_name)
    skeleton_video = list()
    with open(path, 'r') as load_f:
        load_list = json.load(load_f)
        if debug:
            print('Load {} success.'.format(path))
    for sk_dict in load_list:
        skeleton = np.array(sk_dict['skeleton'])
        skeleton_video.append(skeleton)
    return skeleton_video


def make_skeleton_video(dir, video_name):
    path = '{}{}'.format(dir, video_name)
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    skeleton_video = list()
    cap = cv2.VideoCapture(path)
    frame = 0
    while True:
        success, img = cap.read()
        if not success:
            break
        skeleton = get_skeleton(pose, img)
        skeleton_video.append(skeleton)
        frame += 1
        if (frame % 50 == 0):
            print('frame {} success!'.format(frame))

    frame = 0
    for i, skeleton in enumerate(skeleton_video):
        if skeleton[0][0] != 0:
            frame += 1
    if frame <= 10:
        print('From make_skeleton_video: no skeleton detected!')
        return []

    return skeleton_video


def get_skeleton(pose, img):
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

        # calculate dis
        # n1, n2 = skeleton[3], skeleton[16]
        # dis = math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)
        # p = 0.1 / dis
        #
        # for id, joint in enumerate(skeleton):
        #     skeleton[id][:3] *= p

    return skeleton


def skeleton2json(skeleton, dir, file_name):
    skeleton_list = skeleton.tolist()

    output_dict = dict()
    output_dict['skeleton'] = skeleton_list

    if not os.path.exists(dir):
        os.mkdir(dir)

    name_without_suffix = file_name.split('.')[0]
    path = '{}{}.json'.format(dir, name_without_suffix)
    with open(path, "w") as f:
        json.dump(output_dict, f)
        print('write {} success'.format(path))


def skeleton_video2json(skeleton_video, dir, file_name):
    output_list = list()
    if not os.path.exists(dir):
        os.mkdir(dir)

    name_without_suffix = file_name.split('.')[0]
    path = '{}{}.json'.format(dir, name_without_suffix)

    for frame, skeleton in enumerate(skeleton_video):
        skeleton_list = skeleton.tolist()

        output_dict = dict()
        output_dict['frame'] = frame
        output_dict['skeleton'] = skeleton_list

        output_list.append(output_dict)

    with open(path, "w") as f:
        json.dump(output_list, f)
        print('write {} success'.format(path))


def play_skeleton_video(skeleton_video, size=(1080, 1920, 3), middle=True, save=False, save_dir='', file_name='',
                        std=False, std_skv=''):
    name_without_suffix = file_name.split('.')[0]
    save_path = '{}{}_skeleton.mp4'.format(save_dir, name_without_suffix)

    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        fps = 30
        _size = (1920, 1080)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(save_path, fourcc, fps, _size, True)

    if middle:
        skeleton_video = change_video_coordinate(skeleton_video)
        std_skv = change_video_coordinate(std_skv)

    for id, skeleton in enumerate(skeleton_video):
        img = np.ndarray(size, np.uint8)
        if std and id < len(std_skv):
            img = draw_skeleton(img, std_skv[id], middle=middle, show=False, video=True)
        img = draw_skeleton(img, skeleton, middle=middle, show=True, video=True)
        if save:
            videoWriter.write(img)
        cv2.waitKey(1)

    if save:
        print('The video is saved as {}.'.format(save_path))
        videoWriter.release()


def test():
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    img_dir = './test_images/'
    img_name = 'test7.jpg'
    img_output_dir = './test_images/output/'
    video_dir = './test_videos/'
    video_name = '/test2.mp4'
    video_output_dir = './test_videos/output/'

    video_json_dir = './test_videos/output/'
    json_name = 'test2.json'
    img = cv2.imread('{}{}'.format(img_dir, img_name))

    img = cv2.resize(img, None, fx=0.3, fy=0.3)

    sk = get_skeleton(pose, img)
    sk = change_coordinate(sk)
    # img = draw_skeleton(img, sk)


    joints = [[3, 4, 6], [3, 5, 7], [4, 6, 8], [5, 7, 9], [17, 19, 21], [18, 20, 22]]

    h, w, c = img.shape
    color = [(255, 0, 255), (0, 255, 0), (0, 0, 255), (255, 255, 255), (255, 0, 0), (255, 255, 0)]
    # 0-purple 1-green 2-red 3-white 4-blue 5-aqua
    sum = 0
    for i, joint in enumerate(joints):

        bx, by = int(sk[joint[0]][0] * w), int(sk[joint[0]][1] * h)
        ex, ey = int(sk[joint[1]][0] * w), int(sk[joint[1]][1] * h)
        bx, by = int(bx + w / 2), int(by + h / 2)
        ex, ey = int(ex + w / 2), int(ey + h / 2)
        # if skeleton[i[0]][3] > 0.2 and skeleton[i[1]][3] > 0.2:
        cv2.line(img, (bx, by), (ex, ey), color[i], int(h / 300))


        bx, by = int(sk[joint[1]][0] * w), int(sk[joint[1]][1] * h)
        ex, ey = int(sk[joint[2]][0] * w), int(sk[joint[2]][1] * h)
        bx, by = int(bx + w / 2), int(by + h / 2)
        ex, ey = int(ex + w / 2), int(ey + h / 2)
        # if skeleton[i[0]][3] > 0.2 and skeleton[i[1]][3] > 0.2:
        cv2.line(img, (bx, by), (ex, ey), color[i], int(h / 100))



        d1 = dc.calculate_angle(sk[joint[0]][0], sk[joint[0]][1],
                                sk[joint[1]][0], sk[joint[1]][1],
                                sk[joint[2]][0], sk[joint[2]][1], )
        print(i, d1)
        sum += d1
    print(sum / 6)
    cv2.imshow('', img)
    cv2.waitKey(0)
