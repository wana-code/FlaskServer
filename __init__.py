from flask import Flask , request
import config

import torch
from torchvision import models
from PIL import Image

from network.Transformer import Transformer
from test_from_code import transform
from tqdm import tqdm_notebook
import os
import base64
import json
import cv2
import numpy as np

#model load
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device : ',device)
styles = ["Hosoda", "Hayao", "Shinkai", "Paprika"]

models = {}

for style in tqdm_notebook(styles):
    model = Transformer()
    model.load_state_dict(torch.load(os.path.join("./pretrained_models/", style + '_net_G_float.pth')))
    model.eval()
    models[style] = model

def model_style(style):        
    image = Image.open(request.files['file'].stream).convert("RGB")

    load_size = 300
    output = transform(models, style, image , load_size)
    plate = model_style_merge(output, 'dish.jpg')
    tshirt = model_style_merge(output, 't_shirt.jpg')
    ecobag = model_style_merge(output, 'ecobag.png')
    output_images = {"output": output,"tshirt": tshirt,"plate": plate,"ecobag": ecobag}
    
    temp={}
    for i, img_ in output_images.items():
        img = cv2.imencode('.jpg', cv2.cvtColor(np.array(img_), cv2.COLOR_BGR2RGB))[1]
        temp[i]=base64.encodebytes(img).decode('utf-8')
        
    return json.dumps(temp)

def model_style_merge(image, goods):
    dst_path = 'C:\\Users\\HP\\project\\flask_app_cartoon\\merge_goods\\'+goods

    src = np.array(image)
    dst = cv2.imread(dst_path, cv2.IMREAD_UNCHANGED) # 붙임 당할 이미지   
    # png 이미지일 경우 변환
    if len(dst.shape) > 2 and dst.shape[2] == 4:
        #slice off the alpha channel
        dst = dst[:, :, :3]
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    width2, height2, _ = dst.shape

    src = cv2.resize(src, dsize=(int(dst.shape[0]*0.3), int(dst.shape[1]*0.3)), interpolation= cv2.INTER_NEAREST)
    width, height, _ = src.shape
    
    dst[int((width2 - width) / 2):int((width2 - width) / 2 + width),int((height2 - height) / 2):int((height2 - height) / 2 + height)] = src.copy()
    
    return dst

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    @app.route('/Hosoda', methods=['POST'])
    def Hosoda():               
        return model_style(styles[0])

    @app.route('/Hayao', methods=['POST'])
    def hayao():
        return model_style(styles[1])

    @app.route('/Shinkai', methods=['POST'])
    def shinkai():
        return model_style(styles[2])

    @app.route('/Paprika', methods=['POST'])
    def paprika():
        return model_style(styles[3])

    return app

# 터미널  :  set FALSK_APP=패키지이름    set FLASK_DEBUG=true
