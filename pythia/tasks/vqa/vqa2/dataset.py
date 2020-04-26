# Copyright (c) Facebook, Inc. and its affiliates.
import os

import torch
import tqdm

from pythia.common.sample import Sample
from pythia.tasks.base_dataset import BaseDataset
from pythia.tasks.features_dataset import FeaturesDataset
from pythia.tasks.image_database import ImageDatabase
from pythia.utils.distributed_utils import is_main_process
from pythia.utils.general import get_pythia_root
from pythia.utils.text_utils import generate_ngrams_range
from pythia.scripts.features.extract_features_vmb import FeatureExtractor
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from PIL import Image

import cv2
import numpy as np
from os.path import expanduser
import sys

sys.path.append('/home/anandkumar/textvqa/content/detectron2')

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

class VQA2Dataset(BaseDataset):
    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__("vqa2", dataset_type, config)
        imdb_files = self.config.imdb_files

        if dataset_type not in imdb_files:
            raise ValueError(
                "Dataset type {} is not present in "
                "imdb_files of dataset config".format(dataset_type)
            )

        self.imdb_file = imdb_files[dataset_type][imdb_file_index]
        self.imdb_file = self._get_absolute_path(self.imdb_file)
        self.imdb = ImageDatabase(self.imdb_file)

        self.kwargs = kwargs
        self.image_depth_first = self.config.image_depth_first
        self._should_fast_read = self.config.fast_read

        self.use_ocr = self.config.use_ocr
        self.use_ocr_info = self.config.use_ocr_info

        self.detection_model = self._build_detection_model()

        self._use_features = False
        if hasattr(self.config, "image_features"):
            self._use_features = True
            self.features_max_len = self.config.features_max_len
            self._return_info = self.config.get("return_info", True)

            all_image_feature_dirs = self.config.image_features[dataset_type]
            curr_image_features_dir = all_image_feature_dirs[imdb_file_index]
            curr_image_features_dir = curr_image_features_dir.split(",")
            curr_image_features_dir = self._get_absolute_path(curr_image_features_dir)

            self.features_db = FeaturesDataset(
                "coco",
                directories=curr_image_features_dir,
                depth_first=self.image_depth_first,
                max_features=self.features_max_len,
                fast_read=self._should_fast_read,
                imdb=self.imdb,
                return_info=self._return_info,
            )

            print("using image features")

    def _get_absolute_path(self, paths):
        if isinstance(paths, list):
            return [self._get_absolute_path(path) for path in paths]
        elif isinstance(paths, str):
            if not os.path.isabs(paths):
                pythia_root = get_pythia_root()
                paths = os.path.join(pythia_root, self.config.data_root_dir, paths)
            return paths
        else:
            raise TypeError(
                "Paths passed to dataset should either be " "string or list"
            )

    def __len__(self):
        return len(self.imdb)

    def try_fast_read(self):
        # Don't fast read in case of test set.
        if self._dataset_type == "test":
            return

        if hasattr(self, "_should_fast_read") and self._should_fast_read is True:
            self.writer.write(
                "Starting to fast read {} {} dataset".format(
                    self._name, self._dataset_type
                )
            )
            self.cache = {}
            for idx in tqdm.tqdm(
                range(len(self.imdb)), miniters=100, disable=not is_main_process()
            ):
                self.cache[idx] = self.load_item(idx)

    def get_item(self, idx):
        if self._should_fast_read is True and self._dataset_type != "test":
            return self.cache[idx]
        else:
            return self.load_item(idx)

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        current_sample = Sample()

        if "question_tokens" in sample_info:
            text_processor_argument = {"tokens": sample_info["question_tokens"]}
        else:
            text_processor_argument = {"text": sample_info["question"]}

        processed_question = self.text_processor(text_processor_argument)

        current_sample.text = processed_question["text"]
        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = torch.tensor(
                sample_info["image_id"], dtype=torch.int
            )
        else:
            current_sample.image_id = sample_info["image_id"]

        current_sample.text_len = torch.tensor(
            len(sample_info["question_tokens"]), dtype=torch.int
        )

        if self._use_features is True:
            features = self.features_db[idx]
            print("image features")
            print(features["image_feature_0"])
            plt.imshow(features["image_feature_0"].tolist())
            plt.axis("off")
            current_sample.update(features)

        # Add details for OCR like OCR bbox, vectors, tokens here
        current_sample = self.add_ocr_details(sample_info, current_sample)
        # Depending on whether we are using soft copy this can add
        # dynamic answer space
        current_sample = self.add_answer_info(sample_info, current_sample)

        return current_sample

    def add_ocr_details(self, sample_info, sample):
        if self.use_ocr:

            # Preprocess OCR tokens

            home = expanduser("~")


            file_name = sample_info["image_id"]
            #file_base_name = os.path.join(home+"/textvqa/content/pythia/data/open_images/detectron_fix_100/fc6/train",file_name)
            file_base_name = os.path.join(home+"/textvqa/content/","fruits.jpg")
            file_base_name = file_base_name.split(".")[0]
            info_file_base_name = file_base_name + "_info.npy"
            #file_base_name = file_base_name + ".npy"
            file_base_name = file_base_name + ".jpg"
            '''

            print("Feature extractor")
            print(file_base_name)
            print(FeatureExtractor().get_detectron_features_thresh([file_base_name],"fc6",0))
            '''

           # print("Getting detectron features")
            #print(self.get_detectron_features(file_base_name))
            #print(self.get_detectron2_prediction(cv2.imread(file_base_name)))



            ocr_token_list = []

            for ocr_token in generate_ngrams_range(sample_info["ocr_tokens"],(1,4)):
                ocr_token_list.append(ocr_token)

            sample_info["ocr_tokens"] = ocr_token_list


            ocr_tokens = [
                self.ocr_token_processor({"text": token})["text"]
                for token in sample_info["ocr_tokens"]
            ]





            # Get embeddings for tokens
            context = self.context_processor({"tokens": ocr_tokens})
            sample.context = context["text"]
            sample.context_tokens = context["tokens"]
            sample.context_feature_0 = context["text"]
            sample.context_info_0 = Sample()
            sample.context_info_0.max_features = context["length"]

            order_vectors = torch.eye(len(sample.context_tokens))
            order_vectors[context["length"] :] = 0
            sample.order_vectors = order_vectors

        if self.use_ocr_info and "ocr_info" in sample_info:
            sample.ocr_bbox = self.bbox_processor({"info": sample_info["ocr_info"]})[
                "bbox"
            ]

        return sample

    def add_answer_info(self, sample_info, sample):
        if "answers" in sample_info:
            answers = sample_info["answers"]
            answer_processor_arg = {"answers": answers}

            if self.use_ocr:
                answer_processor_arg["tokens"] = sample_info["ocr_tokens"]
            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)

            sample.answers = processed_soft_copy_answers["answers"]
            sample.targets = processed_soft_copy_answers["answers_scores"]

        return sample

    def idx_to_answer(self, idx):
        return self.answer_processor.convert_idx_to_answer(idx)

    def format_for_evalai(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, question_id in enumerate(report.question_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
            else:
                answer = self.answer_processor.idx2word(answer_id)
            if answer == self.context_processor.PAD_TOKEN:
                answer = "unanswerable"

            predictions.append({"question_id": question_id.item(), "answer": answer})

        return predictions

    def _build_detection_model(self):

        cfg.merge_from_file('/home/anandkumar/textvqa/content/model_data/detectron_model.yaml')
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load('/home/anandkumar/textvqa/content/model_data/detectron_model.pth',
                                map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        print("Built Detection model")
        return model

    def get_detectron_features(self, image_path):
        im, im_scale = self._image_transform(image_path)
        img_tensor, im_scales = [im], [im_scale]
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to('cuda')
        with torch.no_grad():
            output = self.detection_model(current_img_list)
        feat_list = self._process_feature_extraction(output, im_scales,
                                                     'fc6', 0.2)
        return feat_list[0]

    def _image_transform(self, image_path):
        path = image_path

        img = Image.open(path)
        im = np.array(img).astype(np.float32)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(800) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > 1333:
            im_scale = float(1333) / float(im_size_max)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)
        return img, im_scale

    def _process_feature_extraction(self, output,
                                    im_scales,
                                    feat_name='fc6',
                                    conf_thresh=0.2):

        print("labels ")
        print(output[0]["labels"].tolist())

        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feat_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]

            max_conf = torch.zeros((scores.shape[0])).to(cur_device)

            for cls_ind in range(1, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                             cls_scores[keep],
                                             max_conf[keep])

            keep_boxes = torch.argsort(max_conf, descending=True)[:100]
            feat_list.append(feats[i][keep_boxes])
        return feat_list

    def get_detectron2_prediction(self,im):
        cfg = get_cfg()
        cfg.merge_from_file("/home/anandkumar/textvqa/content/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
        predictor = DefaultPredictor(cfg)

        # Make prediction
        outputs = predictor(im)

        predictions = outputs["instances"].to("cpu")

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None

        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])



        return self._create_text_labels(classes,scores,metadata.get("thing_classes", None))

    def _create_text_labels(self, classes, scores, class_names):
        """
        Args:
            classes (list[int] or None):
            scores (list[float] or None):
            class_names (list[str] or None):

        Returns:
            list[str] or None
        """
        labels_treshold=[]
        labels = None
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:

                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                for i in len(range(scores)):
                    if scores[i]>80:
                        labels_treshold.append(classes[i])



        return labels_treshold





