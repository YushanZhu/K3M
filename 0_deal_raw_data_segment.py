# -*- coding: utf-8 -*-
from tqdm import tqdm
import requests
import argparse
import os


def clean_and_down_pic(raw_data_path, segment_id):# 清洗数据，下载图片 
    out_file = 'data/id_title_pvs_cls.txt'+str(segment_id) 
    
    with open(raw_data_path, 'r', encoding='UTF-8', errors='ignore') as f_in:
        with open(out_file, 'w') as f_out: 
            item_count = 0
            for line in tqdm(f_in):
                lin = line.strip()
                try:
                    itemID, title, image_url, pv_str, category = lin.strip().split('\t')
                    pv_str = pv_str.replace("#", "")

                    pic = requests.get(image_url)
                    pic_type = image_url.split('.')[-1]
                    #image_url  = 'https://img.alicdn.com/imgextra' + image_url.split('img.alicdn.com/imgextra')[-1]
                    pic_name = str(item_count) + '_s' + str(segment_id) + '.'+ pic_type
                    # print(image_url)
                    # import pdb; pdb.set_trace()
                    if pic.status_code != 200:
                        raise Exception('无图片，pic.status_code != 200')
                    with open(os.path.join('data/image', pic_name),'wb') as fp:
                        fp.write(pic.content)
                    
                except Exception as e:  # 
                    print(e)
                    print(lin)
                    continue
                

                f_out.write('\t'.join([str(item_count)+ '_s'+str(segment_id), title, pic_name, pv_str, category, itemID]) + '\n')
                f_out.flush()
                
                item_count += 1
                
if __name__=='__main__':
    raw_data_path_train ='data/raw_multidata_of_product_preatrain.small_train'
    raw_data_path_valid ='data/raw_multidata_of_product_preatrain.small_valid'

    if not os.path.exists('data/image'):
        os.mkdir('data/image')

    parser = argparse.ArgumentParser()
    # parser.add_argument("--segment_id",default=0,type=int)
    args = parser.parse_args()

    clean_and_down_pic(raw_data_path_train, segment_id = 0)
    clean_and_down_pic(raw_data_path_valid, segment_id = 1)
