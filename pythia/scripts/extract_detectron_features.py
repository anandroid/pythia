import sys

sys.path.append('/home/anandkumar/textvqa/content/detectron2')

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import json
import os
import requests
from pythia.tasks.image_database import ImageDatabase
from PIL import Image
import numpy



def get_detectron2_prediction(im):
    cfg = get_cfg()
    cfg.merge_from_file(
        "/home/anandkumar/textvqa/content/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    predictor = DefaultPredictor(cfg)

    # Make prediction
    outputs = predictor(im)

    predictions = outputs["instances"].to("cpu")

    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None

    bboxes = []

    for ibox in boxes:
        bboxes.append(ibox.tolist())

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    dict_to_save_json = {}
    dict_to_save_json['boxes'] = bboxes
    dict_to_save_json['scores'] = scores.tolist()
    dict_to_save_json['classes'] = classes.tolist()
    dict_to_save_json['labels'] = _create_text_labels(classes.tolist(), scores.tolist(),
                                                      metadata.get("thing_classes", None))

    return dict_to_save_json


def _create_text_labels(classes, scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """

    labels_treshold = []
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:

            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            for i in range(len(scores)):
                if scores[i] > 0:
                    labels_treshold.append(labels[i])

    return labels_treshold


def get_actual_image(image_path):

    if image_path.startswith('http'):
        path = requests.get(image_path, stream=True).raw
    else:
        path = image_path



    return path


def runForFiles():
    path = os.path.join(
        os.path.abspath(__file__),
        # "../../../configs/vqa/textvqa/lorra.yml"
        "../../../pythia/common/defaults/configs/tasks/vqa/textvqa.yml"
    )

    # dir = '../../../pythia/data/imdb/textvqa_0.5/imdb_textvqa_train.npy'
    dir = '/home/anandkumar/textvqa/content/pythia/data/imdb/textvqa_0.5/imdb_textvqa_train.npy'

    imageDataBaseDic = ImageDatabase(dir)

    total = len(imageDataBaseDic)
    count=0

    start=20495
    count=start
    for i in range(start,len(imageDataBaseDic)):
        imageDataElement = imageDataBaseDic[i+start]
        try:
            url = imageDataElement['flickr_300k_url']
            image_id = imageDataElement['image_id']

            dict = {}
            # cv2.imread(get_actual_image(url))
            img = Image.open(get_actual_image(url)).convert('RGB')

            dict = get_detectron2_prediction(numpy.asarray(img))
            #print(dict)
            with open('/home/anandkumar/textvqa/content/pythia/data/detectron_processed/' + image_id + '.json', 'w') as fp:
                json.dump(dict, fp, indent=4)

            count = count + 1
            print("Progress :" + str(count)+"/" + str(total) + " : "+image_id)
            img.close()
        except:
            print("")

    '''
    dir = "../../../data/open_images/resnet152/"

    images_npy = os.listdir(dir)
    '''

    '''
    imdb_file = imdb_files[dataset_type][imdb_file_index]
    imdb_file = _get_absolute_path(imdb_file)
    imdb = ImageDatabase(self.imdb_file)

    dict={}
    dict = get_detectron2_prediction(cv2.imread(file_base_name))
    with open(sample_info['image_id']+'.json', 'w') as fp:
            json.dump(dict, fp,  indent=4)
    '''


runForFiles()
