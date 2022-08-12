from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import pandas as pd
import torch
import numpy as np
import json
# from . import yolov5
import traceback

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.cuda import amp

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import datetime
import base64
from io import BytesIO

from PIL import Image
from torchvision import transforms
from django.conf import settings

class PredictImageFromBase64(APIView):
    def get(self, request):
        out_dict = {}
        out_dict['message'] = '画像データをBase64形式で、POSTしてください。'
        return Response(out_dict, status=status.HTTP_200_OK)
    def post(self, request):
        try:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            input_image = json.loads(request.body)

            # 処理時間の計測開始
            start = time.time()
            print(f"request receive_time: {datetime.datetime.fromtimestamp(start)}")

            ### モデル定義
            # model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
            # model = torch.hub.load('ultralytics/yolov5', 'custom', path='/code/static/model/asahuka96hozon.pt')
            # model.to(device)
            #model = torch.nn.DataParallel(model, device_ids=[0, 1])
            #print(device)
            ### モデル設定
            # model.conf = 0.25  # NMS confidence threshold
            # model.iou = 0.45  # NMS IoU threshold
            # model.agnostic = False  # NMS class-agnostic
            # model.multi_label = False  # NMS multiple labels per box
            # model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
            # model.max_det = 1000  # maximum number of detections per image
            # model.amp = False  # Automatic Mixed Precision (AMP) inference
            
            # model.device = [2,3]
            # model.nproc_per_node = 2

            model = settings.YOLO_MODEL
            print("YOLOv5 🚀 torch 1.10.2+cu102")

            ### 入力
            # inp_im = torch.rand(1, 3, 1280, 640)
            ### バイナリデータ <- base64でエンコードされたデータ
            input_list = []
            num = len(input_image["input_image"])
            for img_base64 in input_image["input_image"]:
                img_binary = base64.b64decode(img_base64)
                img = np.frombuffer(img_binary,dtype=np.uint8)
                img = Image.open(BytesIO(img))
                input_list.append(img)

            results = model(input_list, size=640)
            out_result = {}

            out_result["output_"] = []
            for k in range(num):
                result = results.pandas().xyxy[k]
                result = result[result["class"]==0]
                out_result["output_"].append(result.to_json(orient="records"))
            # out_result["output_"] = [results.pandas().xyxy[k].to_json(orient="records") for k in range(num)]

            # 返却の直前で計測終了と経過時間を表示
            elapsed_time = time.time() - start
            print(f"elapsed_time: {elapsed_time}")

            return Response(out_result["output_"])

        except Exception as e:
            # Error内容がわからないため，Exceptionで全て受け取る
            print(traceback.format_exc())
