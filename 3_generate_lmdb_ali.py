import os
import numpy as np
# from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ
from tensorpack.dataflow import *
import lmdb
import json
import pdb
import csv
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'cls_prob']
import sys
import pandas as pd
import zlib
import base64
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)


def open_tsv(fname):
    id_s=[]
    title_s=[]
    pvs_s=[]
    product_cls_s=[]
    
    print("Opening %s Data File..." % fname)
    with open(fname, 'r') as f:
        for line in tqdm(f):
            item_id,title,pvs,product_cls,_=line.strip().split('\t')#最后是原item_id
            
            id_s.append(item_id)
            title_s.append(title)
            pvs_s.append(pvs)
            product_cls_s.append(product_cls)
            
    df = pd.DataFrame({'item_id':id_s,'caption':title_s,'pvs':pvs_s,'product_cls':product_cls_s})
    #df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df
    
#def _file_name(row):
 #   return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

class Conceptual_Caption(RNGDataFlow):
    """
    """
    def __init__(self, corpus_path, numfile=1,filetype=None, num_caps=3136161,shuffle=False,given_file_id=None):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.shuffle = shuffle
        self.num_file = numfile
        self.name = os.path.join(corpus_path, filetype+'.tsv.%d')
        print(self.name)
        if given_file_id:
            self.infiles = [self.name % i for i in given_file_id]
        else:#没给就是所有的
            self.infiles = [self.name % i for i in range(self.num_file)]
        for index,the_file in enumerate(self.infiles):
            print(index,':',the_file)#文件排个序
        
        self.counts = []
        self.num_caps = num_caps#
        
        self.cap_pv_cls ={}
        
        if filetype=='train':
            all_df=pd.read_csv('./data/image_lmdb_json/df_train.csv',encoding='utf-8',dtype={'image_id': str,'item_ID': str})#指定类型
        elif filetype=='dev':
            all_df=pd.read_csv('./data/image_lmdb_json/df_val.csv',encoding='utf-8',dtype={'image_id': str,'item_ID': str})
            
        for image_id,pv,caption,category in zip(all_df['image_id'],all_df['pv'],all_df['caption'],all_df['category']):
            self.cap_pv_cls[image_id]=(pv,caption,category)
        
        
    def __len__(self):
        return self.num_caps

    def __iter__(self):
        for infile in self.infiles:
            print('infile:',infile)
            count = 0
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in reader:
                    image_id = item['image_id']
                    image_h = item['image_h']
                    image_w = item['image_w']
                    num_boxes = item['num_boxes']
                    
                    boxes = np.frombuffer(base64.b64decode(item['boxes'][2:-1]), dtype=np.float32).reshape(
                        int(num_boxes), 4)
                    
                    features = np.frombuffer(base64.b64decode(item['features'][2:-1]), dtype=np.float32).reshape(
                        int(num_boxes), 2048)
                    
                    cls_prob = np.frombuffer(base64.b64decode(item['cls_prob'][2:-1]), dtype=np.float32).reshape(int(num_boxes), 1601)
                    
                    pv,caption,category=self.cap_pv_cls[image_id]
                    
                    yield [features, cls_prob, boxes, num_boxes, image_h, image_w, image_id, caption, pv, category]

if __name__ == '__main__':
    """train"""#改数据 numfile以及num_caps
    with open('./data/image_lmdb_json/df_train.csv') as f:
        total_train_lines = sum(1 for line in f)
    
    ds = Conceptual_Caption(corpus_path='./data/image_features',filetype='train', num_caps = total_train_lines) 
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    LMDBSerializer.save(ds1, './data/image_lmdb_json/training_feat_all.lmdb')

    
    """validation"""
    with open('./data/image_lmdb_json/df_val.csv') as f:
        total_valid_lines = sum(1 for line in f)
    
    ds = Conceptual_Caption(corpus_path='./data/image_features',filetype='dev', num_caps = total_valid_lines)
    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    LMDBSerializer.save(ds1, './data/image_lmdb_json/validation_feat_all.lmdb')