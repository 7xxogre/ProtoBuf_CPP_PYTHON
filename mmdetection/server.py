import socket
from os import walk
import os
import time
from PIL import Image
import cv2
#Custom Function
from one_image_inference import save_predict_obj_img,\
                                get_predict
                                
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmcv import Config
import glob

class Server:
    def __init__(self, 
                host:str='127.0.0.1', 
                port:int=3070,
                data_path:str = '',
                ini_path:str = os.path.join('D:', 'WATIZ', 'ini', 'AlgaeList.txt'),
                model_path:str=''
                ):
        
        """[Socket Server Settings]

        Args:
            host (str, optional): [host ip number]. Defaults to '127.0.0.1'.
            port (int, optional): [port number]. Defaults to 3070.
            data_path (str, optional): [파이썬 서버로 보낼 원폰사진들 들어있는 폴더경로]. Defaults to ''.
            ini_path (str, optional): [Label Folder List]. Defaults to ''.
            model_path (str, optional): [DL model path]. Defaults to ''.
        """
        self.host = host
        self.port = port
        self.model_path = model_path
        self.data_path = data_path
        self.serv_addr = (self.host, self.port)
        self.ini_path = ini_path
        
        #assert path != None, 'Images Path is Empty'
        #self.path = path
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
        cfg = Config.fromfile("customized_config.py")
        self.model = init_detector(cfg, checkpoint=self.model_path)
        
        print(f'[server]default model AI Server is ready')

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
        # test_sample_path = './init_sample.bmp'
        # _,_ = self.diagnoisis_volt(test_sample_path)
        self.establish()
        self.send_msg('server is ready')
        
        while True:
            start = time.process_time()
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
            
            elif img_path == 'start1':
                result = os.path.join(os.getcwd(), 'result')
                if not os.path.exists(result):
                    os.mkdir(result)
            
                tmp_path = os.path.join(result, 'temp')
                if not os.path.exists(tmp_path):
                    os.mkdir(tmp_path)
                
                for class_name in self.model.CLASSES:
                    if not os.path.exists(os.path.join(result, class_name)):
                        os.mkdir(os.path.join(result, class_name))
                        
                
                data_path_list = [filename for filename in glob.iglob(self.data_path+'/*')
                                    if filename.lower().endswith(('.bmp', 'jpg'))]
                
                for img_path in data_path_list:
                    save_predict_obj_img(self.model, 
                                        img_path, 
                                        save_path=tmp_path)
                    result_list = get_predict(self.model,
                                    img_path)
                    
                    with open(os.path.join(self.data_path, 
                                        f'{img_path[:-3]}txt'), 'w+') as f:
                        for result in result_list:
                            f.write('\n'.join(result))
                    
                    

            end = time.process_time()

        
        self.disconnect()


if __name__ =='__main__':
    host = '127.0.0.1'
    port = 3070
    
    
    S = Server(host, 
            port,
            data_path = os.path.join(os.getcwd(), 'data'),
            ini_path = os.path.join(os.getcwd(), 'etc', 'AlgaeList.txt'),
            model_path=os.path.join('work_dir', 'epoch_50.pth'))
    
    S.server_activate()
    