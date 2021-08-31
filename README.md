# ProtoBuf_CPP_PYTHON

1. mmdetection 폴더에 work_dir 폴더를 생성
2. work_dir 폴더에 학습시킨 pth 파일을 넣어줌
3. mmdetection 폴더에 해당 모델 config 파일을 넣어줌
4. server.py를 열어 적절한 가중치 파일 경로와 config 파일 경로 설정
5. Valveseatinspection.bat 실행
6. python server와 cpp server가 연결된 후 나오는 mfc 창의 button3를 눌러 inferece를 할 data 폴더 선택 후 적용
