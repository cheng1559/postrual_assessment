import socket

if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = ("127.0.0.1", 9999)
    sock.sendto(str('connect').encode(), address)
    while True:
        receive_data, client = sock.recvfrom(9999)
        data = receive_data.decode().split(' ')
        print('receive {}'.format(data))
        if len(data) == 3:
            if data[1] == '+':
                ans = int(data[0]) + int(data[2])
            if data[1] == '-':
                ans = int(data[0]) - int(data[2])
            if data[1] == '*':
                ans = int(data[0]) * int(data[2])
            if data[1] == '/':
                ans = int(data[0]) // int(data[2])
            sock.sendto(str(ans).encode(), address)
        else:
            print('undefined operation: {}'.format(data))
            sock.sendto(str('error').encode(), address)
    sock.close()

