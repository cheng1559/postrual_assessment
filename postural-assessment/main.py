import tools.skeleton_generator as sg
import tools.dtw_calculate as dc
import os

if __name__ == '__main__':
    # sg.test()
    video_dir = './test_videos/'
    video_save_dir = './test_videos/save/'
    json_dir = './test_videos/output/'
    json_std_dir = './test_videos/output/standard/'

    type = 'rufengsibi'
    video_list = os.listdir(video_dir)
    for video_name in video_list:
        if not '.' in video_name:
            continue
        name = video_name.split('.')[0]

        # if name != 'rufengsibi2':
        #     continue

        # video_name = '{}.mp4'.format(name)
        json_name = '{}.json'.format(name)

        # sk_video = sg.make_skeleton_video(video_dir, video_name)
        # sg.skeleton_video2json(sk_video, json_dir, video_name)

        sk_video = sg.read_skeleton_video(json_dir, json_name)

        std_skv = sg.read_skeleton_video(json_std_dir, '{}.json'.format(type))
        # sg.play_skeleton_video(sk_video, (1080, 1920, 3), save=True, middle=True, save_dir=video_save_dir,
        #                        file_name=video_name, std=False, std_skv=std_skv)

        d = dc.evaluate(sk_video, type, show=False) # baihe louxi banlanchui rufengsibi
        score = max(min(-52 * d * d + 2.7 * d + 100.8, 100), 0)
        print(name, d, round(score, 2))


    # dc.test()