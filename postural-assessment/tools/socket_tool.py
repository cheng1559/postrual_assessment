import cv2
import socket
import tools.get_video as gv


def test():
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
            video_output_dir = data[1]
            video_length = int(data[2])
            gv.get_video(video_output_dir, video_length)
            evaluate(video_dir, video_name, pose_type)
            sock.sendto(str('camera success!').encode(), address)

        elif data[0] == 'quit':
            print('quit')
            sock.sendto(str('already quit').encode(), address)
            break

        else:
            print('undefined operation: {}'.format(data))
            sock.sendto(str('error').encode(), address)
    sock.close()
