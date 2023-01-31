import numpy as np
from dtw import dtw
import math
import matplotlib.pyplot as plt
import tools.skeleton_generator as sg


def sk_distance(sk_1, sk_2):
    distance = 0
    # sk_1 = sg.change_coordinate(sk_1)
    # sk_2 = sg.change_coordinate(sk_2)
    for i in range(0, 27):
        node1, node2 = sk_1[i], sk_2[i]
        # print(i, node1, node2)
        distance += math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)
    return distance


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
