import numpy as np
from dtw import dtw
import math
import matplotlib.pyplot as plt
import tools.skeleton_generator as sg


def joint_distance(sk_1, sk_2):
    distance = 0
    # sk_1 = sg.change_coordinate(sk_1)
    # sk_2 = sg.change_coordinate(sk_2)
    weight = [1, 1, 1, # head
              1, 1, 1, # upper body
              2, 2, 2, 1, 1, 1, 5, 5, 5, 5, # arms and hands
              1, 1, 1, # lower body
              1, 1, 1, 1, 5, 5, 5, 5 # legs and feet
              ]
    for i in range(0, 27):
        node1, node2 = sk_1[i], sk_2[i]
        distance += math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2) * weight[i]
    return distance * 0.4


def calculate_angle(x1, y1, x2, y2, x3, y3):
    v1 = [x1 - x2, y1 - y2]
    v2 = [x3 - x2, y3 - y2]
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        # print("Zero magnitude vector!")
        return 0

    vector_dot_product = np.dot(v1, v2)
    arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle = np.degrees(arccos)
    return angle


def angel_distance(sk_1, sk_2):
    distance = 0
    joints = [[3, 4, 6], [3, 5, 7], [4, 6, 8], [5, 7, 9], [17, 19, 21], [18, 20, 22]]
    for i, joint in enumerate(joints):
        d1 = calculate_angle(sk_1[joint[0]][0], sk_1[joint[0]][1],
                             sk_1[joint[1]][0], sk_1[joint[1]][1],
                             sk_1[joint[2]][0], sk_1[joint[2]][1],)
        d2 = calculate_angle(sk_2[joint[0]][0], sk_2[joint[0]][1],
                             sk_2[joint[1]][0], sk_2[joint[1]][1],
                             sk_2[joint[2]][0], sk_2[joint[2]][1],)
        distance += abs(d1 - d2)
        # print(i, abs(d1 - d2))
    return distance / 6


def sk_distance(sk_1, sk_2):
    d1 = joint_distance(sk_1, sk_2)
    # d2 = angel_distance(sk_1, sk_2)
    d2 = 0
    return d1 + d2


def evaluate(eval_skv, name, show=True):
    std_json_dir = './test_videos/output/standard/'
    std_json_name = '{}.json'.format(name)
    std_skv = sg.read_skeleton_video(std_json_dir, std_json_name)

    std_skv = sg.change_video_coordinate(std_skv)
    eval_skv = sg.change_video_coordinate(eval_skv)

    d, cost_matrix, acc_cost_matrix, path = dtw(std_skv, eval_skv, dist=sk_distance)
    if show:
        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        plt.plot(path[0], path[1], 'w')
        plt.show()
    return d / (len(std_skv) + len(eval_skv))


# def test():
#     std_json_dir = './test_videos/output/'
#     std_json_name = 'baihe.json'
#     eval_json_dir = './test_videos/output/'
#     eval_json_name = 'test.json'
#
#     std_sk = sg.read_skeleton_video(std_json_dir, std_json_name)
#     eval_sk = sg.read_skeleton_video(eval_json_dir, eval_json_name)
#
#
#     d, cost_matrix, acc_cost_matrix, path = dtw(std_sk, eval_sk, dist=sk_distance)
#
#     print(d)
#     print(acc_cost_matrix.T)
#
#     plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
#     plt.plot(path[0], path[1], 'w')
#     plt.show()
