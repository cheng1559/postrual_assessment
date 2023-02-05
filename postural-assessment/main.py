import tools.skeleton_generator as sg
import tools.dtw_calculate as dc
import os
import cv2
import tools.get_video as gv
import socket
import tools.socket_tool as st
import time


def evaluate(video_dir, video_name, pose_type, show=True):
    name = video_name.split('.')[0]
    video_save_dir = './test_videos/save/'
    json_std_dir = './test_videos/output/standard/'
    json_dir = './test_videos/output/'

    sk_video = sg.make_skeleton_video(video_dir, video_name)
    std_skv = sg.read_skeleton_video(json_std_dir, '{}.json'.format(pose_type))

    if len(sk_video) == 0:
        print('From main.evaluate: no skeleton detected!')
        return

    sg.skeleton_video2json(sk_video, json_dir, video_name)

    if show:
        sg.play_skeleton_video(sk_video, (1080, 1920, 3), save=True, middle=True, save_dir=video_save_dir,
                               file_name=video_name, std=True, std_skv=std_skv)

    d = dc.evaluate(sk_video, pose_type, show=show)
    score = max(min(-52 * d * d + 2.7 * d + 100.8, 100), 0)
    # print(name, d, round(score, 2))
    return score


def evalutae_from_json(json_dir, video_name, pose_type, show=True):
    name = video_name.split('.')[0]
    json_name = '{}.json'.format(name)
    json_std_dir = './test_videos/output/standard/'
    video_save_dir = './test_videos/save/'

    sk_video = sg.read_skeleton_video(json_dir, json_name)
    std_skv = sg.read_skeleton_video(json_std_dir, '{}.json'.format(pose_type))

    if len(sk_video) == 0:
        print('From main.evaluate_from_json: no skeleton detected!')
        return

    if show:
        sg.play_skeleton_video(sk_video, (1080, 1920, 3), save=False, middle=True, save_dir=video_save_dir,
                               file_name=video_name, std=True, std_skv=std_skv)

    d = dc.evaluate(sk_video, pose_type, show=show)
    score = max(min(-52 * d * d + 2.7 * d + 100.8, 100), 0)
    print(name, d, round(score, 2))
    return score


def test(pose_type):  # baihe louxi banlanchui rufengsibi
    # sg.test()
    video_dir = './test_videos/'
    json_dir = './test_videos/output/'

    video_list = os.listdir(video_dir)
    for video_name in video_list:
        if not '.' in video_name:
            continue
        # evaluate(video_dir, video_name, pose_type)
        evalutae_from_json(json_dir, video_name, pose_type, show=False)


if __name__ == '__main__':
    video_output_dir = './video_output/'
    video_length = 8000
    video_dir = video_output_dir
    video_name = 'video.mp4'
    # video_dir = './test_videos/'
    # video_name = 'banlanchui5.mp4'
    json_dir = './test_videos/output/'
    pose_type = 'banlanchui'

    # gv.get_video(video_output_dir, video_length)


    # evaluate(video_dir, video_name, pose_type)
    # evalutae_from_json(json_dir, video_name, pose_type, show=True)
    # test(pose_type)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = ("127.0.0.1", 9999)
    sock.sendto(str('connect').encode(), address)
    while True:
        receive_data, client = sock.recvfrom(9999)
        data = receive_data.decode().split(' ')
        print('receive {}'.format(data))

        if data[0] == 'test':
            argc = data[1]
            argv = data[2:]

        elif data[0] == 'camera':
            video_length = int(data[1])
            gv.get_video(video_output_dir, video_length)
            sock.sendto(str('camera success!').encode(), address)

        elif data[0] == 'evaluate':
            if len(data) < 3:
                print('undefined operation: {}'.format(data))
                sock.sendto(str('error').encode(), address)
                continue

            video_length = int(data[1])
            pose_type = data[2]
            video_name = 'video.mp4'
            gv.get_video(video_output_dir, video_length)

            # video_dir = './test_videos/'
            # video_name = 'banlanchui5.mp4'

            t = time.perf_counter()
            score = evaluate(video_dir, video_name, pose_type, show=False)
            print(time.perf_counter() - t)
            sock.sendto(str('your score is {}.'.format(score)).encode(), address)

        elif data[0] == 'quit':
            print('quit')
            sock.sendto(str('already quit').encode(), address)
            break

        else:
            print('undefined operation: {}'.format(data))
            sock.sendto(str('error').encode(), address)
    sock.close()