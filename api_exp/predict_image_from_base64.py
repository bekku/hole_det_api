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
import ast

def imgpil_to_npimage(image_pil):
  transform = transforms.Resize((32*19, 32*29))
  image = transform(image_pil)
  image = np.array(image)
  # 色チャネルが4で出力されるため, 最初の3のみ使用する。
  if len(image[0]) != 3:
    image = image[:,:,:3]
  # [0.485, 0.456, 0.406] [0.229, 0.224, 0.225]で正規化するよ〜
  image = image/256
  image =  image - np.array([0.485, 0.456, 0.406])
  image = image/np.array([0.229, 0.224, 0.225])
  image = image.transpose((2,0,1))
  return image


class PredictImageFromBase64(APIView):
    def get(self, request):
        out_dict = {}
        out_dict['message'] = '画像データをBase64形式で、POSTしてください。'
        return Response(out_dict, status=status.HTTP_200_OK)
    def post(self, request):
        try:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            input_image = json.loads(request.body)
            print(f"request receive_time: {datetime.datetime.fromtimestamp(time.time())}")

            ### 入力
            # inp_im = torch.rand(1, 3, 1280, 640)
            ### バイナリデータ <- base64でエンコードされたデータ
            input_list = []
            num = len(input_image["input_image"])
            for img_base64 in input_image["input_image"]:
                img_binary = base64.b64decode(img_base64)
                img = np.frombuffer(img_binary,dtype=np.uint8)
                img = Image.open(BytesIO(img))
                input_list.append(img.convert('RGB'))


            # ===================================== 陥没穴YOLO =====================================
            # 処理時間の計測開始
            yolo_start = time.time()
            # 陥没穴検出yolo modelの定義
            yolo_model = settings.YOLO_MODEL
            print("YOLOv5 🚀 torch 1.10.2+cu102")
            results = yolo_model(input_list, size=640)
            out_result = {}
            out_result["output_"] = []
            for k in range(num):
                result = results.pandas().xyxy[k]
                result = result[result["class"]==0]
                out_result["output_"].append(result.to_json(orient="records"))
            # out_result["output_"] = [results.pandas().xyxy[k].to_json(orient="records") for k in range(num)]

            # 返却の直前で計測終了と経過時間を表示
            yolo_elapsed_time = time.time() - yolo_start
            print(f"yolo_model_elapsed_time: {yolo_elapsed_time}")
            print(out_result["output_"])
            # =================================================================================

            # ===================================== CAOD =====================================
            # 処理時間の計測開始
            caod_all_start = time.time()

            # ① u_plusplusモデルの定義
            # 処理時間の計測開始
            upp_start = time.time()

            upp_model = settings.UPP_MODEL
            img_list = np.array(list(map(imgpil_to_npimage, input_list)))
            x_tensor = torch.from_numpy(img_list).to(device).float()
            pr_mask = upp_model.predict(x_tensor)        
            # 画像1枚時 : (3, H, W), 複数時(N, 3, H, W)であるため, 1枚の時はunsqueezeする。
            if num==1:
                pr_mask = torch.unsqueeze(pr_mask, 1)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())

            to_yolo_imagePIL = []
            for enunum, mask_im in enumerate(pr_mask):
                mask_im = mask_im.transpose((1, 2, 0))
                # mask_imを白黒画像にする。
                for i in range(3):
                    mask_im[:, :, i] = mask_im[:, :, 1]

                image_pil = input_list[enunum]
                transform = transforms.Resize((32*19, 32*29))
                image = transform(image_pil)
                image = np.array(image)
                # 色チャネルが4で出力されるため, 最初の3のみ使用する。
                if len(image[0]) != 3:
                    image = image[:,:,:3]

                view_image = image*mask_im
                view_image = Image.fromarray(view_image.astype(np.uint8))
                to_yolo_imagePIL.append(view_image)

            # 返却の直前で計測終了と経過時間を表示
            upp_elapsed_time = time.time() - upp_start
            print(f"U++model_elapsed_time: {upp_elapsed_time}")

            # ② CAODモデルの定義
            # 処理時間の計測開始
            caod_start = time.time()
            
            caod_model = settings.CAOD_MODEL
            input_list_tocaod = to_yolo_imagePIL

            results_caod = caod_model(input_list_tocaod, size=640*2)
            num = len(input_list_tocaod)
            out_result_caod = {}
            out_result_caod["output_"] = []
            for k in range(num):
                result_caod = results_caod.pandas().xyxy[k]
                result_caod["class"]=0
                result_caod["name"]="object"
                out_result_caod["output_"].append(result_caod.to_json(orient="records"))

            # 返却の直前で計測終了と経過時間を表示
            caod_elapsed_time = time.time() - caod_start
            print(f"caod_model_elapsed_time: {caod_elapsed_time}")
            
            # ③ 既知物体の排除機構
            # 処理時間の計測開始
            remove_model_start = time.time()

            # 検出された物体のpillow型の画像リスト作成
            object_images_list = []
            for i in range(len(out_result_caod["output_"])):
                for i_bbox in eval(out_result_caod["output_"][i]):
                    i_bbox_image = input_list_tocaod[i].crop((i_bbox["xmin"], i_bbox["ymin"], i_bbox["xmax"], i_bbox["ymax"]))
                    object_images_list.append(i_bbox_image)

            # 学習済みモデルで検出された物体画像から、取り除き対象のみ結果として保持
            if len(object_images_list)!=0:
                yolo_to_remove_model = settings.YOLO_TO_REMOVE_MODEL
                results_to_remove = yolo_to_remove_model(object_images_list, size=300)

                objects_out_result = []
                num_to_remove = len(object_images_list)
                for k in range(num_to_remove):
                    result_to_remove = results_to_remove.pandas().xyxy[k]
                    result_to_remove = result_to_remove[
                        (
                            (result_to_remove["name"]=="bicycle")|
                            (result_to_remove["name"]=="car")| 
                            (result_to_remove["name"]=="person")|
                            (result_to_remove["name"]=="bus")|
                            (result_to_remove["name"]=="truck")
                        )
                    ]
                    objects_out_result.append(result_to_remove.to_json(orient="records"))

                # objectの名称が取り除き対象ではない時、returnするリストに追加
                out_result_after_remove = [[] for i in range(len(out_result_caod["output_"]))]
                object_number = -1
                for i in range(len(out_result_caod["output_"])):
                    for i_bbox in eval(out_result_caod["output_"][i]):
                        object_number +=1
                        if len(objects_out_result[object_number])==2:
                            out_result_after_remove[i].append(i_bbox)
            else:
                out_result_after_remove = [[] for i in range(len(out_result_caod["output_"]))]

            # josn形式に変更
            # out_result_after_remove = json.dumps(out_result_after_remove)
            for i in range(len(out_result_after_remove)):
                out_result_after_remove[i] = str(out_result_after_remove[i])
                out_result_after_remove

            # 返却の直前で計測終了と経過時間を表示
            remove_model_elapsed_time = time.time() - remove_model_start
            print(f"remove_model_elapsed_time: {remove_model_elapsed_time}")
            # =================================================================================

            # 陥没穴YOLOとCAODモデルの結果結合
            return_result_outputs = [[] for i in range(len(out_result["output_"]))]
            yolo_result_ast = out_result["output_"]
            caod_result_ast = out_result_after_remove
            for i in range(len(yolo_result_ast)):
                return_result_outputs[i] = json.dumps(ast.literal_eval(yolo_result_ast[i]) + ast.literal_eval(caod_result_ast[i]))

            # 返却の直前で計測終了と経過時間を表示
            caod_all_elapsed_time = time.time() - caod_all_start
            print(f"caod_all_elapsed_time: {caod_all_elapsed_time}")

            return Response(return_result_outputs)

        except Exception as e:
            print(traceback.format_exc())
