import tools.skeleton_generator as sg
import tools.dtw_calculate as dc
import os
import cv2
import tools.get_video as gv


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
    print(name, d, round(score, 2))
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
    gv.get_video(video_output_dir, video_length)

    video_dir = video_output_dir
    video_name = 'video.mp4'
    # video_dir = './test_videos/'
    # video_name = 'banlanchui5.mp4'
    json_dir = './test_videos/output/'
    pose_type = 'banlanchui'

    evaluate(video_dir, video_name, pose_type)
    # evalutae_from_json(json_dir, video_name, pose_type, show=True)
    # test(pose_type)