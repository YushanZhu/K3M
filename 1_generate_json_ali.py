import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import csv
from multiprocessing import Process
import random
import json
import pdb
import pandas as pd
import random
import zlib


import os
import io
import glob 
import math
import re
import pandas as pd

#import detectron2

"""# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
"""
# import some common libraries
import numpy as np
import torch
import glob
from tqdm import tqdm

"""
NUM_OBJECTS = 36

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, \
    fast_rcnn_inference_single_image

"""
def write_json(file,data):
    f=open(file,"w",encoding="utf-8")
    json.dump(data,f,indent=2,ensure_ascii=False)
    return



    
def load_image_ids(raw_file_path):
    id_ids = []
    id_titles =[]
    id_picnames = []
    id_pvs =[]
    id_itemIDs =[]
    id_categorys = []
    with open(raw_file_path, 'r') as f:
        for line in tqdm(f):
           
            image_id,title, pic_name, pvs,category,item_ID=line.strip().split('\t') # 最后是原item_id
            id_ids.append(image_id)
           
            id_titles.append(title)
            id_picnames.append(pic_name)
            id_pvs.append(pvs)
            id_itemIDs.append(item_ID)
            id_categorys.append(category)

    df=pd.DataFrame({'image_id':id_ids,'caption':id_titles,'pic':id_picnames,'pv':id_pvs,'itemID':id_itemIDs,'category':id_categorys})
    return df


def generate_df(train_or_val):##json文件产自所有文件，而提取图片特征文件不需要
    
    if train_or_val=='val':#1 
        all_raw_file_path=['./data/id_title_pvs_cls.txt1'] # 1号是valid
    elif train_or_val=='train':
        file_num = len(glob.glob('./data/id_title_pvs_cls.txt*'))
        print('file_num:',file_num) #0 (2 3 4 5 6 7 8 9 ..)
        all_raw_file_path = ['./data/id_title_pvs_cls.txt'+str(i) for i in range(file_num)]
        all_raw_file_path.remove('./data/id_title_pvs_cls.txt1')
        
    print(all_raw_file_path)
    

    all_df=None
    for index,infile in enumerate(all_raw_file_path):
        print(index,':',infile)
        this_df = load_image_ids(infile)
        
        if index==0:
            all_df=this_df
        else:
            all_df=pd.concat([all_df,this_df])
        
    
    all_df.to_csv("./data/image_lmdb_json/df_"+train_or_val+".csv",encoding = "utf-8")
    print(all_df)
    
    
def generate_json(train_or_val):
    all_df=pd.read_csv("./data/image_lmdb_json/df_"+train_or_val+".csv",encoding = "utf-8")
    print(all_df)
    for this_obj in ['caption','pic','pv','itemID','category']: 
        this_json=[]
        for image_id, value in zip(all_df['image_id'], all_df[this_obj]):
            this_json.append((image_id,value))
        write_json("./data/image_lmdb_json/"+this_obj+"_"+train_or_val+".json",this_json)
    
    
if __name__ == '__main__':
    if not os.path.exists('data/image_lmdb_json'):
        os.mkdir('data/image_lmdb_json')

    generate_df('val')
    generate_json('val')
    
    generate_df('train')
    generate_json('train')