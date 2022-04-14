# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import logging
import math
import os
import random

import lmdb
import numpy as np
import tensorpack.dataflow as td

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import sys
import pdb

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

MAX_MSGPACK_LEN = 1000000000

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = (
            (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).reshape(1, K)

    anchors_area = (
            (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).reshape(N, 1)

    boxes = np.repeat(anchors.reshape(N, 1, 4), K, axis=1)
    query_boxes = np.repeat(gt_boxes.reshape(1, K, 4), N, axis=0)

    iw = (
            np.minimum(boxes[:, :, 2], query_boxes[:, :, 2])
            - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
            + 1
    )
    iw[iw < 0] = 0

    ih = (
            np.minimum(boxes[:, :, 3], query_boxes[:, :, 3])
            - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1])
            + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def deserialize_lmdb(ds):
    return msgpack.loads(
        ds[1],
        raw=False,
        max_bin_len=MAX_MSGPACK_LEN,
        max_array_len=MAX_MSGPACK_LEN,
        max_map_len=MAX_MSGPACK_LEN,
        max_str_len=MAX_MSGPACK_LEN,
    )


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
            self,
            image_feat=None,
            image_target=None,
            caption=None,
            is_next=None,

            pv=None,  # add
            is_next_pv_v=None,
            is_next_pv_t=None,

            lm_labels=None,

            lm_labels_pv=None,

            image_loc=None,
            num_boxes=None,
            overlaps=None,
    ):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.image_feat = image_feat
        self.caption = caption
        self.is_next = is_next  # nextSentence

        self.pv=pv#add
        self.is_next_pv_v=is_next_pv_v
        self.is_next_pv_t = is_next_pv_t

        self.lm_labels = lm_labels  # masked words for language model

        self.lm_labels_pv=lm_labels_pv  # add masked words for language model of pv

        self.image_loc = image_loc
        self.image_target = image_target
        self.num_boxes = num_boxes
        self.overlaps = overlaps


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids=None,
            input_mask=None,
            segment_ids=None,
            is_next=None,
            lm_label_ids=None,

            input_ids_pv=None,
            input_mask_pv=None,
            segment_ids_pv=None,
            lm_label_ids_pv=None,
            is_next_pv_v=None,
            is_next_pv_t=None,

            image_feat=None,
            image_target=None,
            image_loc=None,
            image_label=None,
            image_mask=None,
            masked_label=None,
            index_p=None,
            index_v=None
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids

        self.input_ids_pv = input_ids_pv
        self.input_mask_pv = input_mask_pv
        self.segment_ids_pv = segment_ids_pv
        self.is_next_pv_v = is_next_pv_v
        self.is_next_pv_t = is_next_pv_t
        self.lm_label_ids_pv = lm_label_ids_pv

        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_target = image_target
        self.image_mask = image_mask
        self.masked_label = masked_label
        self.index_p = index_p
        self.index_v = index_v


class ConceptCapLoaderTrain_struc(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the
            GPU, which is faster).
    """

    def __init__(
            self,
            corpus_path,
            tokenizer,
            bert_model,
            seq_len,
            seq_len_pv=None,
            encoding="utf-8",
            visual_target=0,
            hard_negative=False,
            batch_size=512,
            shuffle=False,
            num_workers=25,
            cache=10000,
            drop_last=False,
            cuda=False,
            local_rank=-1,
            objective=0,
            visualization=False,
            train_lmdb_file="training_feat_all.lmdb",
    ):
        TRAIN_DATASET_SIZE = 3119449

        if dist.is_available() and local_rank != -1:

            num_replicas = dist.get_world_size()
            rank = dist.get_rank()

            lmdb_file = os.path.join(
                corpus_path, "training_feat_part_" + str(rank) + ".lmdb"
            )
        else:
            lmdb_file = os.path.join(corpus_path, train_lmdb_file)
            # lmdb_file = os.path.join(corpus_path, "validation_feat_all.lmdb")

            print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        ds = td.LocallyShuffleData(ds, cache)
        caption_path = os.path.join(corpus_path, "caption_train.json")
        pv_path = os.path.join(corpus_path, "pv_train.json")
        # caption_path = os.path.join(corpus_path, "caption_val.json")

        preprocess_function = BertPreprocessBatch(
            caption_path,
            pv_path,
            tokenizer,
            bert_model,
            seq_len,
            36,
            self.num_dataset,
            seq_len_pv=seq_len_pv,
            encoding="utf-8",
            visual_target=visual_target,
            objective=objective,
        )

        ds = td.PrefetchData(ds, 5000, 1)
        ds = td.MapData(ds, preprocess_function)
        # self.ds = td.PrefetchData(ds, 1)
        ds = td.PrefetchDataZMQ(ds, num_workers)
        # print("batch_size",batch_size)
        self.ds = td.BatchData(ds, batch_size)
        # self.ds = ds
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):

        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, lm_label_ids, is_next, input_ids_pv, input_mask_pv, segment_ids_pv, lm_label_ids_pv, is_next_pv_v, is_next_pv_t,image_feat, image_loc, image_target, image_label, image_mask, masked_label, index_p, index_v, image_id = (
                batch
            )


            batch_size = input_ids.shape[0]

            sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
            sum_count[sum_count == 0] = 1
            g_image_feat = np.sum(image_feat, axis=1) / sum_count
            image_feat = np.concatenate(
                [np.expand_dims(g_image_feat, axis=1), image_feat], axis=1
            )
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(
                np.array([[0, 0, 1, 1, 1]], dtype=np.float32), batch_size, axis=0
            )
            image_loc = np.concatenate(
                [np.expand_dims(g_image_loc, axis=1), image_loc], axis=1
            )

            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            batch = (
                input_ids,
                input_mask,
                segment_ids,
                lm_label_ids,
                is_next,

                input_ids_pv, input_mask_pv, segment_ids_pv, lm_label_ids_pv, is_next_pv_v, is_next_pv_t,

                image_feat,
                image_loc,
                image_target,
                image_label,
                image_mask,
            )


            yield tuple([torch.tensor(data) for data in batch] + [index_p, index_v, image_id])

    def __len__(self):
        return self.ds.size()


class ConceptCapLoaderVal_struc(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the
            GPU, which is faster).
    """

    def __init__(
            self,
            corpus_path,
            tokenizer,
            bert_model,
            seq_len,
            seq_len_pv=None,
            encoding="utf-8",
            visual_target=0,
            batch_size=512,
            shuffle=False,
            num_workers=25,
            cache=5000,
            drop_last=False,
            cuda=False,
            objective=0,
            visualization=False,
    ):
        lmdb_file = os.path.join(corpus_path, "validation_feat_all.lmdb")
        caption_path = os.path.join(corpus_path, "caption_val.json")
        pv_path = os.path.join(corpus_path, "pv_val.json")

        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        preprocess_function = BertPreprocessBatch(
            caption_path,
            pv_path,
            tokenizer,
            bert_model,
            seq_len,
            36,
            self.num_dataset,
            seq_len_pv=seq_len_pv,
            encoding="utf-8",
            visual_target=visual_target,
            visualization=visualization,
            objective=objective,
        )

        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):
        for batch in self.ds.get_data():
            
            input_ids, input_mask, segment_ids, lm_label_ids, is_next, input_ids_pv, input_mask_pv, segment_ids_pv, lm_label_ids_pv, is_next_pv_v, is_next_pv_t,image_feat, image_loc, image_target, image_label, image_mask, masked_label, index_p, index_v, image_id = (
                batch
            )


            batch_size = input_ids.shape[0]
            sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
            sum_count[sum_count == 0] = 1
            g_image_feat = np.sum(image_feat, axis=1) / sum_count
            image_feat = np.concatenate(
                [np.expand_dims(g_image_feat, axis=1), image_feat], axis=1
            )
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(
                np.array([[0, 0, 1, 1, 1]], dtype=np.float32), batch_size, axis=0
            )
            image_loc = np.concatenate(
                [np.expand_dims(g_image_loc, axis=1), image_loc], axis=1
            )

            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            batch = (
                input_ids,
                input_mask,
                segment_ids,
                lm_label_ids,
                is_next,

                input_ids_pv, input_mask_pv, segment_ids_pv, lm_label_ids_pv, is_next_pv_v, is_next_pv_t,

                image_feat,
                image_loc,
                image_target,
                image_label,
                image_mask,
            )

            yield tuple([torch.tensor(data) for data in batch] + [index_p, index_v, image_id])

    def __len__(self):
        return self.ds.size()


class BertPreprocessBatch(object):
    def __init__(
            self,
            caption_path,
            pv_path,
            tokenizer,
            bert_model,
            seq_len,
            region_len,
            data_size,
            seq_len_pv=None,
            split="Train",
            encoding="utf-8",
            visual_target=0,
            visualization=False,
            objective=0,
    ):

        self.split = split
        self.seq_len = seq_len
        self.seq_len_pv = seq_len_pv
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.visual_target = visual_target
        self.num_caps = data_size
        self.captions = []#[i[1] for i in json.load(open(caption_path, "r"))]
        self.pvs = []#[i[1] for i in json.load(open(pv_path, "r"))]
        self.pvs_len = len(self.pvs)
        self.captions_len = len(self.captions)
        # self.captions = list(json.load(open(caption_path, "r")).values())
        self.visualization = visualization
        self.objective = objective
        self.bert_model = bert_model

    def __call__(self, data):

        image_feature_wp, image_target_wp, image_location_wp, num_boxes, image_h, image_w, image_id, caption, pv, category= (
            data
        )

        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_target = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_location = np.zeros((self.region_len, 5), dtype=np.float32)

        # calculate the IOU here.
        overlaps = iou(image_location_wp, image_location_wp)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        image_target[:num_boxes] = image_target_wp
        image_location[:num_boxes, :4] = image_location_wp

        image_location[:, 4] = (
                (image_location[:, 3] - image_location[:, 1])
                * (image_location[:, 2] - image_location[:, 0])
                / (float(image_w) * float(image_h))
        )

        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        if self.visual_target == 0:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_target)
        else:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_feature)

        label, label_pv_t, label_pv_v=0,0,0
        tokens_caption = self.tokenizer.encode(caption)
        if len(pv.strip())>0 and (not pv.strip().endswith(';')):
            pv=pv+';'
        tokens_pv = self.tokenizer.encode(pv)
        #print(pv)
        #print(tokens_pv)
        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=tokens_caption,
            is_next=label,

            pv=tokens_pv,  # add
            is_next_pv_v=label_pv_v,
            is_next_pv_t=label_pv_t,

            image_loc=image_location,
            num_boxes=num_boxes,
            overlaps=overlaps,
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(
            cur_example, self.seq_len, self.tokenizer, self.region_len, pv, self.seq_len_pv
        )

        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.is_next,

            cur_features.input_ids_pv,
            cur_features.input_mask_pv,
            cur_features.segment_ids_pv,
            cur_features.lm_label_ids_pv,
            cur_features.is_next_pv_v,
            cur_features.is_next_pv_t,

            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_target,
            cur_features.image_label,
            cur_features.image_mask,
            cur_features.masked_label,
            cur_features.index_p,
            cur_features.index_v,
            image_id,
        )
        return cur_tensors

    def random_cap(self, caption):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """

        if self.visualization:
            return caption, 0

        if self.objective != 2 and random.random() > 0.5:
            caption = self.get_random_caption()
            label = 1
        else:
            label = 0

        return caption, label
    def random_cap_pv(self, caption,pv):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        if self.visualization:
            return caption, pv, 0,0,0

        if self.objective != 2 :
            prob=random.random()
            if prob < 0.1:
                caption = self.get_random_caption()#
                return caption, pv, 1, 1, 0
            elif prob < 0.2:
                pv = self.get_random_pv()
                return caption, pv, 0, 1, 1
            elif prob < 0.3:
                pv = self.get_random_pv()
                caption = self.get_random_caption()
                return caption, pv, 1, 1, 1
            else:
                return caption, pv, 0, 0, 0
        else:
            return caption, pv, 0, 0, 0
        


    def get_random_caption(self):
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # add the hard negative mining objective here.
        rand_doc_idx = random.randint(0, self.captions_len - 1)
        caption = self.captions[rand_doc_idx]

        return caption
    def get_random_pv(self):
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # add the hard negative mining objective here.
        rand_doc_idx = random.randint(0, self.pvs_len - 1)
        caption = self.pvs[rand_doc_idx]

        return caption

    def convert_example_to_features(
            self, example, max_seq_length, tokenizer, max_region_length, pv=None,  max_seq_length_pv=None
    ):
        """
        """
        image_feat = example.image_feat
        tokens = example.caption

        tokens_pv = example.pv#add

        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)
        is_next = example.is_next

        is_next_pv_v=example.is_next_pv_v#add
        is_next_pv_t = example.is_next_pv_t

        overlaps = example.overlaps

        self._truncate_seq_pair(tokens, max_seq_length - 2)
        self._truncate_seq_pair(tokens_pv, max_seq_length_pv - 2)

        tokens, tokens_label = self.random_word(tokens, tokenizer, is_next)
        tokens_pv, tokens_label_pv = self.random_word_pv(tokens_pv, tokenizer, is_next_pv_v)
        

        image_feat, image_loc, image_label, masked_label = self.random_region(
            image_feat, image_loc, num_boxes, is_next, overlaps
        )

        # concatenate lm labels and account for CLS, SEP, SEP
        lm_label_ids = [-1] + tokens_label + [-1]
        lm_label_ids_pv = [-1] + tokens_label_pv + [-1]
        # print(tokens)
        tokens = tokenizer.add_special_tokens_single_sentence(tokens)
        tokens_pv = tokenizer.add_special_tokens_single_sentence(tokens_pv) # [cls_token_id] + token_ids + [sep_token_id].
        
        index_p, index_v = self.index_pv(tokens_pv) #

        segment_ids = [0] * len(tokens)
        segment_ids_pv = [0] * len(tokens_pv)

        input_ids = tokens  # tokenizer.convert_tokens_to_ids(tokens)
        input_ids_pv = tokens_pv  # tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        input_mask_pv = [1] * (len(input_ids_pv))
        image_mask = [1] * (num_boxes)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        while len(input_ids_pv) < max_seq_length_pv:
            input_ids_pv.append(0)
            input_mask_pv.append(0)
            segment_ids_pv.append(0)
            lm_label_ids_pv.append(-1)
        
        assert len(index_p) == len(index_v)#一样长才能对应
        
        while len(index_p) < 10:#最多10个
            index_p.append([0,0])
        while len(index_v) < 10:#最多10个
            index_v.append([0,0])
            
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length

        assert len(input_ids_pv) == max_seq_length_pv
        assert len(input_mask_pv) == max_seq_length_pv
        assert len(segment_ids_pv) == max_seq_length_pv
        assert len(lm_label_ids_pv) == max_seq_length_pv

        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            lm_label_ids=np.array(lm_label_ids),
            is_next=np.array(example.is_next),

            input_ids_pv=np.array(input_ids_pv),
            input_mask_pv=np.array(input_mask_pv),
            segment_ids_pv=np.array(segment_ids_pv),
            lm_label_ids_pv=np.array(lm_label_ids_pv),
            is_next_pv_v=np.array(example.is_next_pv_v),
            is_next_pv_t=np.array(example.is_next_pv_t),


            image_feat=image_feat,
            image_target=image_target,
            image_loc=image_loc,
            image_label=np.array(image_label),
            image_mask=np.array(image_mask),
            masked_label=masked_label,
            index_p = np.array(index_p),
            index_v = np.array(index_v)
        )
        return features

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()

    def random_word(self, tokens, tokenizer, is_next):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability

            # if is_next == 1 and self.objective != 0:
            #     prob = 1 # not sample mask
            if prob < 0.15 and (not self.visualization):
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(len(tokenizer))
                    # torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label
    
    def index_pv(self, tokens):
        # [cls_token_id] + 真token_ids + [sep_token_id].
        #[cls, 6132, 7305, 6199, 131, 2861, 7216, 132, xxx, xxx, 131, xxx, xxx, xxx, 132...
        index_131=[]#[4,10]
        index_132=[]#[7,14]
        for i,tok_id in enumerate(tokens):# : ;
            if tok_id==131:
                index_131.append(i)#:
            if tok_id==132:
                index_132.append(i)#;
        if len(index_132)==len(index_131):#
            pass
        elif len(index_132)==len(index_131)-1:#
            index_131=index_131[:-1]#
        else: #
            index_131=[]
            index_132=[]
        
        index_p=[]
        index_v=[]
        pv_begin=1
        for idx131, idx132 in zip(index_131, index_132):
            index_p.append([pv_begin, idx131])#[[1,4],[8,10]]
            index_v.append([idx131+1, idx132])#[[5,7],[11,14]]
            pv_begin = idx132+1#8
        
        return index_p, index_v 
            
        
            
    def random_word_pv(self, tokens, tokenizer, is_next):
        #[6132, 7305, 6199, 131, 2861, 7216, 132, xxx, xxx, 131, xxx, xxx, xxx, 132...
        #131与132之间是v
        
        index_131=[]
        index_132=[]
        for i,tok_id in enumerate(tokens):#
            if tok_id==131:
                index_131.append(i)
            if tok_id==132:
                index_132.append(i)
        
        #print('index_131',index_131)
        #print('index_132',index_132)
        #print('len(tokens)',len(tokens))
        
        if len(index_132)==len(index_131)-1: 
            index_132.append(len(tokens)) 
        if len(index_132)>1: 
            index_132=index_132[1:]#
            index_131=index_131[1:]#
            
        output_label = [-1]*len(tokens)
        #print(tokens)
        for beg_i,end_i in zip(index_131,index_132):#v index
            for i in range(beg_i+1,end_i):# v
                output_label[i]=tokens[i]
                tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        
        return tokens, output_label
            
        
        have_131=-1
        have_132=-1
        try:
            have_131=tokens.index(131)#3
            have_132=tokens.index(132)#6#need mask 4(have_131+1),5  for i in range(have_131+1,have_132):mask
        except: 
            have_132=len(tokens)
            
            
        output_label = [-1]*len(tokens)
        if have_131!=-1 and have_132!=-1: 
            for i in range(have_131+1,have_132):
                output_label[i]=tokens[i] 
                tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
            return tokens, output_label
        
    
        output_label = []
        for i, token in enumerate(tokens):
            #print(tokens)
            prob = random.random()
            # mask token with 15% probability

            # if is_next == 1 and self.objective != 0:
            #     prob = 1 # not sample mask
            if prob < 0.15 and (not self.visualization):
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(len(tokenizer))
                    # torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc, num_boxes, is_next, overlaps):
        """
        """
        output_label = []
        masked_label = np.zeros((image_feat.shape[0]))  # 全是补齐了的36个
        max_length = len(masked_label)  # 36

        
        if num_boxes < max_length:
            zero_array = np.zeros((num_boxes, max_length - num_boxes))
            overlaps = np.column_stack((overlaps, zero_array))

        for i in range(num_boxes):
            prob = random.random()
            # mask token with 15% probability

            # if is_next == 1 and self.objective != 0:
            #     prob = 1 # if the target is inaligned mask, then not sample mask
            if prob < 0.15 and not self.visualization:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0
                # mask the overlap regions into zeros
                # print('concept_cap_dataset overlaps.py\n',overlaps.shape)
                # print('\t\t\tmasked_label',len(masked_label))
                # print('\t\t\tnum_boxes',num_boxes)
                # print('\t\t\timage_feat.shape',image_feat.shape)
                # print('\t\t\tmasked_label1',masked_label)
                # print('\t\t\toverlaps[i]',overlaps[i])
                # print('\t\t\toverlaps.shape',overlaps.shape)
                # masked_label = np.logical_or(masked_label, overlaps[i] > 0.4)#modify
                """上一句长度报错，可以overlaps[i]后补0：和0PADDING的关系肯定也是0吧，
                与其他图关系大于0.4的mask,这样理解，每个box都累计一次"""
                masked_label = np.logical_or(masked_label, overlaps[i] > 0.4)  # modify
                # print('\t\t\tmasked_label2',masked_label)

                # 10% randomly change token to random token
                # elif prob < 0.9:
                # tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return image_feat, image_loc, output_label, masked_label
