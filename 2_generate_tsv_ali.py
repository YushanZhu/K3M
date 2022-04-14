import base64
import numpy as np
import cv2
import csv
from multiprocessing import Process
import json
import pandas as pd
import zlib
import os

from tqdm import tqdm

import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# import some common libraries
import numpy as np
import cv2
import torch
import glob

NUM_OBJECTS = 36

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, \
    fast_rcnn_inference_single_image

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features', 'cls_prob']

MIN_BOXES = 36
MAX_BOXES = 36


def read_json(file):
    f=open(file,"r",encoding="utf-8").read()
    return json.loads(f)

def write_json(file,data):
    f=open(file,"w",encoding="utf-8")
    json.dump(data,f,indent=2,ensure_ascii=False)
    return


def open_tsv(fname, folder):
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption", "filename"], usecols=range(0, 2))
    df['folder'] = folder
    print("Processing", len(df), " Images:")
    return df

def write_to_tsv(output_path: str, file_columns: list, data: list):
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(output_path, "w", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=file_columns, dialect='tsv_dialect')
        writer.writerows(data)
    csv.unregister_dialect('tsv_dialect')

def read_from_tsv(file_path: str, column_names: list) -> list:
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(file_path, "r") as wf:
        reader = csv.DictReader(wf, fieldnames=column_names, dialect='tsv_dialect')
        datas = []
        for row in reader:
            data = dict(row)
            datas.append(data)
    csv.unregister_dialect('tsv_dialect')
    return datas


def _file_name(row):
    return "%s/%s" % (row['folder'], (zlib.crc32(row['url'].encode('utf-8')) & 0xffffffff))

    
    


def get_detections_from_image(predictor,raw_image,image_id):
    with torch.no_grad():
        raw_height,raw_width=raw_image.shape[:2]
        # print("original image size: ",raw_height,raw_width)

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        # print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        # print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape) # 154 x 4

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        # print('Pooled features size:', feature_pooled.shape) # 154 x 2048

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        # print("outputs: ",outputs)
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:],
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS: # 36个框
                break
        # 由800 x 800 换成 736 x 736
        instances = detector_postprocess(instances, raw_height, raw_width) # 恢复出原来的图片大小
        roi_features = feature_pooled[ids].detach() # 36 x 2048
        #print(roi_features)
        #print('roi_features:',roi_features.size())# 36 * 2048
        selected_probs=probs[ids]
        #print(selected_probs)
        #print("selected_probs: ",selected_probs.size())# 36 * 1601
        
        #print('instances:',instances.pred_boxes.tensor.cpu().numpy())# 36 * 4
        

        if torch.sum(torch.isnan(roi_features))>0:
            return None

        return_data={
            "image_id":image_id,
            "image_h":raw_height,
            "image_w":raw_width,
            "num_boxes":len(ids),
            "boxes":base64.b64encode(instances.pred_boxes.tensor.cpu().numpy()),
            "features":base64.b64encode(roi_features.cpu().numpy()),
            "cls_prob":base64.b64encode(selected_probs.cpu().numpy())
        }

    return return_data

def get_predictor():
    cfg = get_cfg()
    cfg.merge_from_file("./py-bottom-up-attention/configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weight
    cfg.MODEL.WEIGHTS = "./faster-rcnn-pkl/faster_rcnn_from_caffe.pkl"
    predictor = DefaultPredictor(cfg)
    print("predictor: ",predictor)

    return predictor



def generate_tsv(image_ids,outfile):
    predictor=get_predictor()

    tsvfile=open(outfile,"w")
    writer=csv.DictWriter(tsvfile,delimiter="\t",fieldnames=FIELDNAMES)

    for (image_id, image_file) in tqdm(image_ids):
        image=cv2.imread('./data/image/' + image_file)
        # print('./data/image/' + image_file)
        
        #if True:
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            detection_feature=get_detections_from_image(predictor,image_rgb,image_id)
            #print(detection_feature)
            if detection_feature==None:
                continue
            writer.writerow(detection_feature)
        except Exception as e:
            print("image_id: ", image_id, image_file)
            #print("error: ", e)
            pass

        # if cnt==5:
        #     break
        # cnt+=1

    return

def read_tsv(tsv_path):
    def correct_pad(origStr):
        #print(origStr)
        if(len(origStr)%2 == 1): 
            origStr += "=="
        elif(len(origStr)%3 == 2): 
            origStr += "="
        return origStr
    csv.field_size_limit(500 * 1024 * 1024)#每个字段大小,否则报错超过指定大小
    with open(tsv_path,'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t",fieldnames=FIELDNAMES)
        #print(reader)
        for row in reader:
            #print(row)
            image_id=row['image_id']
            image_h=row["image_h"]
            image_w=row["image_w"]
            num_boxes=row["num_boxes"]
            #从第3个字符取，最后一个不要，因为存的格式是: b'xxx', 只有中间的xxx有用
            boxes = np.frombuffer(base64.b64decode(row['boxes'][2:-1]), dtype=np.float32).reshape(int(num_boxes), 4)
            features = np.frombuffer(base64.b64decode(row['features'][2:-1]), dtype=np.float32).reshape(int(num_boxes), 2048)
            cls_prob = np.frombuffer(base64.b64decode(row['cls_prob'][2:-1]), dtype=np.float32).reshape(int(num_boxes), 1601)
            """print(image_id)
            print(image_h)
            print(image_w)
            print(num_boxes)
            print(boxes)
            print(features)
            print(cls_prob)
            """

def get_train(tsv_id = 0):#36900000 tsv_id 0-40
    print('tsv_id:',tsv_id)
    """train"""
    train_image_ids= read_json('./data/image_lmdb_json/pic_train.json')
    counts_in_each_tsv=900000 
    this_image_ids=train_image_ids[counts_in_each_tsv*tsv_id:counts_in_each_tsv*(tsv_id+1)]
    #for this_id in this_image_ids:
     #   print(this_id)
    generate_tsv(this_image_ids,"./data/image_features/train.tsv."+str(tsv_id))
    
def get_valid(tsv_id = 0): # tsv_id 0-9
    print('tsv_id:',tsv_id)
    """validation"""
    dev_image_ids = read_json('./data/image_lmdb_json/pic_val.json')
    counts_in_each_tsv=100000 #10万数据写一个文件
    this_image_ids=dev_image_ids[counts_in_each_tsv*tsv_id:counts_in_each_tsv*(tsv_id+1)]
    #for this_id in this_image_ids:
     #   print(this_id)
    generate_tsv(this_image_ids, "./data/image_features/dev.tsv."+str(tsv_id))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    if not os.path.exists('data/image_features'):
        os.mkdir('data/image_features')

    get_train(tsv_id=0)#0-40 41个
    get_valid(tsv_id=0)#0-8 9个
    
    