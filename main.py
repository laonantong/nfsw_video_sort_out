import numpy as np
import cv2
from nsfw_classifier import Classifier
from every_thing_sdk import search
import time
import logging


video_extensions_list = ["mp4", "avi", "wmv", "mov",
                         "mkv", "m2ts", "mpg", "rmvb", "vob", "flv"]
classifier = Classifier()
thres = 0.99
interval = 64

if __name__ == "__main__":
    video_files = []
    for exten in video_extensions_list:
        video_files.extend(search("*."+exten))

    for video in video_files:
        video_path = video[0]
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("can not open")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print("can not get frame")
        print("总帧数 = ", total_frames)
        select_frames = []
        start_time = time.time()
        for i in range(0, total_frames, (total_frames+63)//interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = frame.astype(np.float32)
                select_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)/255.0)
        end_time = time.time()
        print(len(select_frames))
        print("抽帧耗时：", end_time - start_time, "秒")

        start_time = time.time()
        pred_result = classifier.pred_nsfw(select_frames, batch_size=16)
        end_time = time.time()
        print("检测耗时：", end_time - start_time, "秒")

        cnt = 0
        for i, pred in enumerate(pred_result):
            if pred > thres:
                cnt += 1
        print("涩涩比例: %f%%" % (cnt*100/len(select_frames)))
