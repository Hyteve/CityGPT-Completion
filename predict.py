from cog import BasePredictor, Path, Input, File
import torch
from generate.model.models_mage import MAGECityPolyGen
from generate.model.pospred_model import MAGECityPosition_Minlen
import os
from generate.generate import main
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import sys
from PIL import Image
sys.path.append('..')
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
import cv2
from shapely.geometry import Polygon
import json

def in_poly_idx(poly, discre = 100):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """

    points = torch.cat([torch.arange(100).repeat_interleave(100).unsqueeze(-1), torch.arange(100).repeat(100).unsqueeze(-1)], dim = -1)
    index = -torch.ones(100*100)
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        condition1 =  (min(y1, y2) < points[:, 1])* (points[:, 1]<= max(y1, y2))  # find horizontal edges of polygon
        x = x1 + (points[:, 1] - y1) * (x2 - x1) / (y2 - y1)
        condition2 = x > points[:, 0]
        index = index*((-(condition1*condition2).long())*2+1)

    out = torch.where(index>0, 0, 1)
    return out 

def pad_poly(poly_list, max_l = 20):
    num = len(poly_list)
    pad_poly = torch.zeros([num, 20, 2])
    pos_in = torch.zeros([num, 2])
    for i in range(num):
        pad_poly[i, :poly_list[i].shape[0], :] = poly_list[i]
        pos_in[i,:] = torch.mean(poly_list[i], dim = 0)//10
    
    return pad_poly.unsqueeze(0), pos_in.unsqueeze(0)

def infgen(model, modelpos, samples, pos, sample_inter, remain_flag, device, discard_prob = 0.5, use_sample=False,finetune = False):

    inlen = samples.shape[1]  
    print("in_len:", inlen)
    samples_iter = samples.clone()
    pos_iter = pos.clone()

    remain_flag = remain_flag.flatten(0,1)
        

    endflag = 0
    gen_iter = []

    num_poly = inlen
    
    for npoly in range(250 - inlen):
        # predition 1
        pred = modelpos(samples_iter, pos_iter, None, generate = True)
        prob_pred = torch.sigmoid(pred[0, :])
        prob_pred = torch.where(prob_pred < discard_prob, torch.tensor(0.0).to(device), prob_pred) 
        while True:
            prob_pred = prob_pred*remain_flag
            if torch.max(prob_pred) < discard_prob:
                endflag = 1
                break      
            if use_sample == False:
                idx_iter = torch.argmax(prob_pred)
            else:            
                idx_iter = torch.multinomial(prob_pred, 1).squeeze(0)
                                
            remain_flag[idx_iter] = 0

            pred_pos = torch.cat([idx_iter.unsqueeze(0)//100, idx_iter.unsqueeze(0)%100],dim=0).unsqueeze(0).unsqueeze(0)
            
            #prediction 2
            predpoly = model.infgen(samples_iter, pos_iter, pred_pos)[0]

            polygon = Polygon(np.array(predpoly[0].clone().detach().cpu()))
            intersect_flag = 0
            for pe in range(samples_iter.shape[1]):
                polyexist = [] 
                for k in range(samples_iter.shape[2]):
                    if samples_iter[0, pe, k, 0] != 0:
                        point = samples_iter[0, pe, k, :].clone().detach().cpu().numpy()
                        polyexist.append(point)
                polyexist = Polygon(polyexist)
                if polygon.intersects(polyexist):
                    intersect_flag = 1
                    break

            for polyexist in sample_inter:
                polyexist = Polygon(polyexist.cpu().numpy())
                if polygon.intersects(polyexist):
                    intersect_flag = 1
                    break

            if intersect_flag == 0:
                break

        if endflag == 1:
            break
        
        poly_add, pos_add = pad_poly(predpoly)
        poly_add = poly_add.to(device)
        pos_add = pos_add.to(device)
        samples_iter = torch.cat([samples_iter, poly_add], dim = 1).detach()
        pos_iter = torch.cat([pos_iter, pos_add], dim = 1).detach()
        num_poly+=1

        gen_iter.append(predpoly.squeeze(0).detach())
        remain_flag = remain_flag*in_poly_idx(predpoly.squeeze(0).detach().cpu()).to(device)

    if finetune:
        gen_iter = model.infgen(samples, pos, pos_iter[:, inlen:])
        gen_iter = [genpoly.squeeze(0).detach() for genpoly in gen_iter]

    return gen_iter


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device('cuda')
        current_path = os.path.dirname(__file__)
        torch.backends.cudnn.benchmark = True

        # Initialize and load the first model
        self.model = MAGECityPolyGen(
            drop_ratio=0.1,
            num_heads=8,
            device=self.device,
            max_build=250,
            depth=12,
            embed_dim=512,
            decoder_embed_dim=512,
            discre=100,
            decoder_depth=8,
            decoder_num_heads=8
        )
        pretrained_model = torch.load(current_path + '/generate/12_8_8_512_512_0.1_1e-3_1000meter.pth')
        self.model.load_state_dict(pretrained_model)
        self.model.to(self.device)
        self.model.eval()

        # Initialize and load the position model
        self.modelpos = MAGECityPosition_Minlen(
            drop_ratio=0.0,
            num_heads=8,
            device=self.device,
            depth=6,
            embed_dim=512,
            decoder_embed_dim=16,
            discre=100,
            patch_num=10,
            patch_size=10,
            decoder_depth=3,
            decoder_num_heads=8,
            pos_weight=100
        )
        pretrained_modelpos = torch.load(current_path + '/generate/128_5000_6_3_8_512_16_0.1_1000meter.pth')
        self.modelpos.load_state_dict(pretrained_modelpos)
        self.modelpos.to(self.device)
        self.modelpos.eval()
    

    
    def predict(self,
            input: str = Input(description="Input to the model")
    ) -> File:
        """Run a single prediction on the model"""
        # ... pre-processing ...
        json_path = './004.json'
        box_start_point = [800, 300]
        
        with open(json_path, 'r') as infile:
            json_data = json.load(infile)
            
        data = json_data.get('scence')

        x, y = box_start_point
        sample_draw_in = []
        sample_draw = []
        for poly in data['scence']:#scenes
            poly = np.array(poly)
            if ((x<poly[:, 0])*(poly[:, 0]<x+1000)*(y<poly[:, 1])*(poly[:, 1]<y+1000)).all():
                sample_draw_in.append(poly)
            else:
                sample_draw.append(poly)

        img = np.ones((2000,2000,3),np.uint8)*255

        leng_prev = len(sample_draw)


        start_point = [(0, 0),
                        (1000, 0),
                        (0, 1000),
                        (1000, 1000),
                        (x, y)]
        
        sample_draw = [torch.tensor(sample) for sample in sample_draw]

        for k in range(0,5):
            sx, sy = start_point[k]
            remain_flag = torch.ones([100,100]).to(self.device)
            if x-sx >= 0:    
                remain_flag[:(x-sx)//10, :] = 0
            else:
                remain_flag[(x-sx)//10:, :] = 0     
            if y-sy >= 0:    
                remain_flag[:, :(y-sy)//10] = 0
            else:
                remain_flag[:, (y-sy)//10:] = 0
                
            sample_in = []
            sample_inter = []
            for poly in sample_draw:
                if ((sx<poly[:, 0])*(poly[:, 0]<sx+1000)*(sy<poly[:, 1])*(poly[:, 1]<sy + 1000)).all():
                    poly_ = poly.clone()
                    poly_[:, 0] -= sx
                    poly_[:, 1] -= sy
                    sample_in.append(poly_)
                elif ((sx<poly[:, 0])*(poly[:, 0]<sx+1000)*(sy<poly[:, 1])*(poly[:, 1]<sy + 1000)).any():
                    poly_ = poly.clone()
                    poly_[:, 0] -= sx
                    poly_[:, 1] -= sy
                    sample_inter.append(poly_)

            sample_in, pos_in = pad_poly(sample_in)
            if sample_in.shape[1]>=250:
                continue

            sample_in = sample_in.to(self.device)

            pos_in = pos_in.to(self.device)
            gen_poly = infgen(model = self.model, modelpos = self.modelpos, samples = sample_in, pos = pos_in, remain_flag = remain_flag, sample_inter = sample_inter, device = self.device, use_sample=False)
            for polybulid in gen_poly:
                polydraw = polybulid.clone()
                polydraw[:, 0] += sx
                polydraw[:, 1] += sy
                sample_draw.append(polydraw)
        
            img = np.ones((2000,2000,3),np.uint8)*255

            for num, poly in enumerate(sample_draw):
                pts = np.array(poly.cpu(), np.int32)
                pts = pts.reshape((-1,1,2)).astype(int)
                # if num<leng_prev:
                #     cv2.fillPoly(img, [pts], color=(238, 159, 153))
                #     cv2.polylines(img,[pts],True,(0,0,0),1)
                # else:
                #     cv2.fillPoly(img, [pts], color=(255, 255, 0))
                #     cv2.polylines(img,[pts],True,(0,0,0),1)
            
                cv2.fillPoly(img, [pts], color=(238, 159, 153))
                cv2.polylines(img,[pts],True,(0,0,0),1)
            
        cv2.imwrite('generate.jpg',img)

        pillow_image = Image.open('generate.jpg')



        # ... post-processing ...
        return File(pillow_image)