import socket
from os import walk
import os
import time
from PIL import Image
import cv2

class Server:
    def __init__(self, 
                host='127.0.0.1', 
                port=3070,
                model=''
                ):
        """[summary]

        Args:
            host (str, optional): [host ip number]. Defaults to '127.0.0.1'.
            port (int, optional): [port number]. Defaults to 3070.
            model (str, optional): [DL model path]. Defaults to ''.
        """
        self.host = host
        self.port = port
        self.model = model
        self.serv_addr = (self.host, self.port)
        
        self.conn_sock = None
        self.dataloader = None
        self.sock = None

    def establish(self):
        """[sever 시작]
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(self.serv_addr) # ip:self.host , port: self.port 
        self.sock.listen(5)
        print('[server]server start')
        self.conn_sock, _ = self.sock.accept()
        print('[Server]server & client Connect')

    def load_model(self):
        # TODO
        """[모델의 가중치 및 하이퍼파라미터 값 로드] 
        model initialize
        
        setting model pth 
        """
        
        print(f'[server]default model ' + 'ganomaly' +' is ready')

    def send_msg(self, msg):
        """[연결된 소켓으로 클라이언트에 메세지 전달]

        Args:
            msg ([type]): [description]
        """
        msg = msg.encode()
        self.conn_sock.sendall(msg)

    
    def recv_msg(self, size = 1024):
        """[클라이언트로 부터 메세지 수령]

        Args:
            size (int, optional): [description]. Defaults to 1024.
            

        Returns:
            [type]: [description]
        """
        msg = self.conn_sock.recv(size)
        if not msg:
            self.conn_sock.close()
            exit()
        return msg.decode()

    def disconnect(self):
        self.conn_sock.close()
        self.sock.close()

    def server_activate(self):
        self.load_model()
        test_sample_path = './init_sample.bmp'
        _,_ = self.diagnoisis_volt(test_sample_path)
        self.establish()
        self.send_msg('server is ready')
        
        while True:
            print('Waiting for transmission....')
            img_path = self.recv_msg()

            print(f'from client: {img_path}\n')
            # msg_data = img_path.split('?')

            # img_path = msg_data[1]
            print('[server]img path(or quit) is : ' + img_path)
            if not img_path:
                print('NO Path')
                continue

            if img_path == 'finish':
                self.disconnect()
                break
            
            start = time.process_time()
            save_path, result = self.diagnoisis_volt(img_path)
            if save_path is None:
                continue

            print(f'result_code {result}\n result_path {save_path}')
            end = time.process_time()
            result_msg = f'{result}?{save_path}'
            self.send_msg(result_msg)
            print(result_msg)
        
        self.disconnect()


if __name__ =='__main__':
    host = '127.0.0.1'
    port = 3070
    S = Server(host, port)
    S.server_activate()
    