import os
import re
import sys
import json
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
from torch import nn
from PIL import Image
from PIL.Image import Image as ImageType
from typing import Tuple, Dict, List

import datasets.transforms as T
from main_synthetic import build_model_main
from datasets import build_dataset
from util import box_ops
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from models.dino.dino import DINO, PostProcess

"""
perform line inference using 2 inputs: image files
and bounding boxes extracted from those images and saved as a JSON.

NOTE: json bbox structure for one image is:
>>> {
...   "images": [{
...       "file_name": "name of image file (without path)",
...       "height": \d+,
...       "width": \d+
...   }],
...   "annotations": [
...       {
...           "bbox": [
...               [l,t,r,b],  # bounding box of the polygon
...           ],
...           "category_id": 1,
...           "image_id": 0,
...           "id": \d+  # ID of the polygon
...       },
...       ...
...   ]
... }
"""

# -------------------------------------------------------
# paths

CWD_PATH = os.getcwd()  # directory from which the cli is run
DIR_PATH = os.path.abspath(os.path.dirname(__file__))  # directory of this file
MODEL_CONFIG_PATH = os.path.join(DIR_PATH, "config", "HWDB_full.py")
MODEL_CHARSET_PATH = os.path.join(DIR_PATH, "data", "dante", "labels_icdar.pkl")
MODEL_CHECKPOINT_PATH = os.path.join(DIR_PATH, "logs", "dante", "medieval_checkpoint.pth")
COCO_PATH = os.path.join(DIR_PATH, "comp_robot", "cv_public_dataset", "COCO2017")

# -------------------------------------------------------
# utils

def to_abspath(p:str|os.PathLike) -> os.PathLike:
    if os.path.isabs(p):
        return os.path.abspath(p)
    else:
        return os.path.abspath(os.path.join(CWD_PATH, p))

# remove extension from filename
def rmext(s:str) -> str:
    return re.sub(r"\.[^\.]+", "", s)

# filenames have either the structure:
#  - wit<witId>_man<manId>_<pageNum>.<extension>
#  - or
# => extract wit<witId>_man<manId>
pattern_aikon = re.compile(r"^wit\d+_man\d+")  # manuscripts from the aikon platform
pattern_oxford = re.compile(r"^[A-Z0-9]+")     # maniscripts sent from oxford
def wid_from_filename(f:str) -> str|None:
    wid = re.search(pattern_aikon, f) or re.search(pattern_oxford, f)
    return wid[0] if wid is not None else None

def img_name_from_json(fp_json:os.PathLike) -> str:
    with open(fp_json, mode="r") as fh:
        data = json.load(fh)
    return data["images"][0]["file_name"]

# the json bbox contains the name of the image file it is related to. test that we can find the file from the json
def test_json_to_img_link(fp_json:os.PathLike, inimg_dir:os.PathLike) -> bool:
    img_name = img_name_from_json(fp_json)
    return os.path.isfile(os.path.join(inimg_dir, img_name))

def to_out_visualization(img_name:str, output_dir:os.PathLike) -> os.PathLike:
    return os.path.join(output_dir, "visualization", f"{rmext(img_name)}.jpg")

def to_out_coco(img_name:str, output_dir:os.PathLike) -> os.PathLike:
    subfolder = wid_from_filename(img_name)
    return os.path.join(output_dir, subfolder, f"{rmext(img_name)}.json")

# -------------------------------------------------------
# i/o

def sanitize(inimg_dir:str, inbbox_dir:str, outimg_dir:str) -> Tuple[os.PathLike]:
    inimg_dir_abs = to_abspath(inimg_dir)
    inbbox_dir_abs = to_abspath(inbbox_dir)
    outimg_dir_abs = to_abspath(outimg_dir)

    if not os.path.isdir(inimg_dir_abs):
        raise FileNotFoundError(f"input image directory '{inimg_dir}' not found. exiting (absolute path: '{inimg_dir_abs}')")
    if not os.path.isdir(inbbox_dir_abs):
        raise FileNotFoundError(f"input bounding box directory '{inbbox_dir_abs}' not found. exiting (absolute path: '{inbbox_dir_abs}')")

    # check that we can create a relationship from the bboxes json to the image files
    assert all(
        test_json_to_img_link(
            os.path.join(inbbox_dir_abs, inbbox),
            inimg_dir_abs
        ) is True
        for inbbox in os.listdir(inbbox_dir_abs)
        if re.search(r"\.json$", inbbox)
    ), f"not all bounding boxes (in '{inbbox_dir_abs}') have a matching image file (in '{inimg_dir_abs}')"

    return inimg_dir_abs, inbbox_dir_abs, outimg_dir_abs


# returns a list of (json_file, img_file) for each json to process
def get_file_pairs(inimg_dir:os.PathLike, inbbox_dir:os.PathLike, sample:bool=False) -> List[Tuple[os.PathLike, os.PathLike]]:
    fp_json_list = [
        os.path.join(inbbox_dir, fn)
        for fn in os.listdir(inbbox_dir)
        if re.search(r"\.json$", fn) is not None
    ]
    file_pairs = [
        (fp_json, os.path.join(inimg_dir, img_name_from_json(fp_json)))
        for fp_json in fp_json_list
    ]
    return file_pairs[:10] if sample and len(file_pairs) > 10 else file_pairs


# create output directory: output_img_dir/<wid>/ (in `output_img_dir`, one directotry per `wid`)
def create_output_structure(inimg_dir:os.PathLike, outimg_dir:os.PathLike) -> None:
    if not os.path.isdir(outimg_dir):
        os.makedirs(outimg_dir)
    witnesses = []
    for f in os.listdir(inimg_dir):
        if f in [".gitignore", ".gitkeep"]:
            continue
        wid = wid_from_filename(f)
        if wid is not None and wid not in witnesses:
            witnesses.append(wid)
        elif wid is None:
            raise ValueError(f"DTLR.pipeline_inference.create_output_structure: could not extract witness ID from filename '{f}'")
    for wid in set(witnesses):
        wid_path = os.path.join(outimg_dir, wid)
        if not os.path.isdir(wid_path):
            os.makedirs(wid_path)
    vis_path = os.path.join(outimg_dir, "visualization")
    if not os.path.isdir(vis_path):
        os.makedirs(vis_path)
    return


# create a visualization and write it to file
def write_visualization(
    fp_out: os.PathLike,
    image: ImageType,
    list_lt: List[Tuple[int,int]],
    list_bbox: List[torch.FloatTensor],
    list_labels: List[List[str]]
) -> None:

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    bbox_image = []
    label_image = []
    for left_top, bbox, labels in zip(list_lt, list_bbox, list_labels):
        for bbb in bbox:
            x, y, w, h = bbb
            x_shifted = x + left_top[0]
            y_shifted = y + left_top[1]
            rect = patches.Rectangle((x_shifted, y_shifted), w, h, linewidth=0.11, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Shift des bboxes pour avoir position relatives dans images
            bbox_image.append(torch.tensor([x_shifted, y_shifted, w, h]).long())
        label_image.append(labels)

    plt.savefig(fp_out, dpi=300)
    return


# format bboxes to coco and write to file
def write_coco(
    fp_out: os.PathLike,
    fp_img_basename: str,
    image: ImageType,
    list_lt: List[Tuple[int,int]],
    list_bbox: List[torch.FloatTensor],
    list_labels: List[List[str]]
) -> None:

    bbox_image = []
    label_image = []
    for left_top, bbox, labels in zip(list_lt, list_bbox, list_labels):
        for bbb in bbox:
            x, y, w, h = bbb
            x_shifted = x + left_top[0]
            y_shifted = y + left_top[1]
            # Shift des bboxes pour avoir position relatives dans images
            bbox_image.append(torch.tensor([x_shifted, y_shifted, w, h]).long())
        label_image.append(labels)

    # convert to coco format and save json
    coco_format = {
        'images': [{'file_name': fp_img_basename, 'id': 0, 'height': image.size[1], 'width': image.size[0]}],
        'annotations': [],
        'categories': []
    }
    # flatten labels
    label_image = [item for sublist in label_image for item in sublist]
    # if there are annotations in the image, build the output coco
    if len(bbox_image):
        bbox_image = torch.stack(bbox_image)

        category_map = {}
        for i, (bbox, label) in enumerate(zip(bbox_image, label_image)):
            # Add category to the categories list if it doesn't exist
            if label not in category_map:
                category_id = len(category_map) + 1  # Assign a new ID to this category
                category_map[label] = category_id
                coco_format['categories'].append({'id': category_id, 'name': label, 'supercategory': 'none'})

            # Add the annotation with the correct category ID
            coco_format['annotations'].append({
                'id': i,
                'image_id': 0,
                'bbox': [bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()],
                'category_id': category_map[label]
            })
    with open(fp_out, mode='w') as fh:
        json.dump(coco_format, fh)
    return


# -------------------------------------------------------
# model

def load_model() -> Tuple[List[str], DINO, Dict[str, PostProcess]]:

    torch.serialization.add_safe_globals([argparse.Namespace])

    device = "cuda:0"
    args = SLConfig.fromfile(MODEL_CONFIG_PATH)
    args.device = device
    args.CTC_training = False
    args.CTC_loss_coef = 0.25
    args.coco_path = ""  # the path of coco
    args.fix_size = False

    with open(MODEL_CHARSET_PATH, mode="rb") as fh:
        labels = pickle.load(fh)
    #NOTE several fonts are available in the charset. we pick the font `all_multi`
    # all available fonts in `charset`:
    # ['antiqua', 'bastarda', 'fraktur', 'gotico-antiqua', 'italic', 'rotunda', 'schwabacher', 'textura', 'all', 'all_multi']
    # for font in labels["charset"]:
    #     print(font, len(labels["charset"][font]))
    charset = labels["charset"]["all_multi"]
    args.charset = charset
    charset_size = len(args.charset)

    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location='cpu')
    features_dim = model.class_embed[0].weight.data.shape[1]

    # 2nd `new_class_embed` is nn.Linear (a linear transform)
    new_class_embed = nn.Linear(features_dim, charset_size, )
    new_decoder_class_embed = nn.Linear(features_dim, charset_size, )
    new_enc_out_class_embed = nn.Linear(features_dim, charset_size, )

    # always true in our case => redefines `new_class_embed`
    if model.dec_pred_class_embed_share:
        class_embed_layerlist = [
            new_class_embed
            for i in range(model.transformer.num_decoder_layers)
        ]

    # 2nd  `new_class_embed` is nn.ModuleList (6 linear layers, stacked)
    new_class_embed = nn.ModuleList(class_embed_layerlist)

    model.class_embed = new_class_embed.to(device)
    model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
    model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)

    model.label_enc = nn.Embedding(charset_size + 1, features_dim).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    return charset, model, postprocessors


# -------------------------------------------------------
# inference pipeline

transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# perform crop of the image + turn that crop into a tensor
def img_crop_to_tensor(
    image:ImageType, bbox_crop: List[float]#l:float, t:float, r:float, b:float
) -> Tuple[float, float, ImageType, torch.FloatTensor, Tuple[int,int], Tuple[int,int]]:
    l, t, r, b = bbox_crop
    l, t, r, b = l-10, t-10, r+10, b+3    # raphael shift
    # l, t, r, b = l-10, t-5, r+10, b+3   # my shift
    crop_image = image.crop((l, t, r, b))
    crop_image_size = crop_image.size
    crop_tensor, _ = transform(crop_image, None)
    crop_tensor_size = crop_tensor.shape[2], crop_tensor.shape[1]
    return l, t, crop_image, crop_tensor, crop_image_size, crop_tensor_size

# run character inference on a single bounding box
# `boxes` = character bounding boxes in `bbox`, in dimensions relative to the tensor
# `final_boxes` = character bounding boxes in `bbox`, in dimensions relative to the actual image
#NOTE are the dimensions of `boxes` and `final_boxes` in (left, top, right, bottom)
# or in `(x, y, height, width)`
def inference(
    model: DINO,
    postprocessors: Dict[str, PostProcess],
    charset: List[str],
    crop_tensor: torch.FloatTensor,
    crop_image_size: Tuple[float,float],
    crop_tensor_size: Tuple[float,float]
) -> List[str]:

    # perform inference
    with torch.no_grad():
        output = model.cuda()(crop_tensor[None].cuda())
        polygones:torch.FloatTensor = output['pred_boxes']
        postprocessors['bbox'].nms_iou_threshold = 0.2
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

        # boxes = all character bounding boxes in `bbox`
        boxes: torch.FloatTensor = output['boxes']
        scores: torch.FloatTensor = output['scores']
        labels: torch.IntTensor = output['labels']
        select_mask: torch.BoolTensor = scores > 0.1

        boxes_xyxy = boxes.clone()
        # each box is now represented by [x,y,width,height]
        boxes = box_ops.box_xyxy_to_cxcywh(boxes)
        boxes = boxes[select_mask]
        scores = scores[select_mask]
        # extract a list of labels for characters in `bbox` and convert to utf-8
        labels = labels[select_mask]
        box_label = [charset[i] for i in labels]
        box_label = [
            bytes(_string, "utf-8").decode("unicode_escape")
            for _string in box_label
        ]

    # remove bounding boxes whose label is `" "` (aka, don't detect spaces)
    #NOTE this also deletes a good amount to other characters so we disable
    #NOTE there is probably an issue with label detection (many chars labels as spaces when they are not spaces)
    # idx_no_spaces = []        # array of indexes to keep
    # box_label_no_spaces = []  # clean labels
    # for i, char in enumerate(box_label):
    #     if char != " ":
    #         idx_no_spaces.append(i)
    #         box_label_no_spaces.append(char)
    # boxes = boxes[idx_no_spaces]
    # box_label = box_label_no_spaces

    # shift bounding boxes from tensor dimension to the OG image's dimension
    ratios_h, ratios_w = tuple(
        float(sz) / float(sz_orig)
        for sz, sz_orig
        in zip(crop_image_size, crop_tensor_size)
    )
    w, h = crop_tensor_size
    final_bboxes = boxes.cpu() * torch.tensor([w, h, w, h])  #
    final_bboxes[:, :2] -= final_bboxes[:, 2:] / 2
    final_bboxes *= torch.Tensor([ratios_w, ratios_h, ratios_w, ratios_h])
    return box_label, final_bboxes


def pipeline(
    model: DINO,
    postprocessors: Dict[str, PostProcess],
    charset: List[str],
    fp_json: os.PathLike,
    fp_img: os.PathLike,
    output_dir: os.PathLike,
    visualize: bool
) -> None:

    list_bbox: List[torch.FloatTensor] = []  # list of torch.FloatTensor for character bounding boxes in the annotation. 1 tensor = bboxes for 1 line. each bbox is structured as [x,y,w,h]
    list_lt: List[Tuple[int,int]] = []       # list of (left,top)
    list_labels: List[List[str]] = []        # list of predicted characters for a bbox

    image_basename = os.path.basename(fp_img)
    image = Image.open(fp_img).convert("RGB")
    with open(fp_json, mode="r") as fh:
        data = json.load(fh)

    for annotation_line in data['annotations']:
        bbox_crop = annotation_line['bbox']
        try:
            l, t, crop_image, crop_tensor, crop_image_size, crop_tensor_size = img_crop_to_tensor(image, bbox_crop)
            list_lt.append((l, t))
            box_label, bbox = inference(
                model,
                postprocessors,
                charset,
                crop_tensor,
                crop_image_size,
                crop_tensor_size
            )
            list_labels.append(box_label)
            list_bbox.append(bbox)
        except Exception as e:
            print(e)
            print('`pipeline()`: Error processing image', fp_img)
            raise

    # convert to output formats and write to file
    if visualize:
        fp_out = to_out_visualization(image_basename, output_dir)
        write_visualization(fp_out, image, list_lt, list_bbox, list_labels)
    else:
        fp_out = to_out_coco(image_basename, output_dir)
        write_coco(fp_out, image_basename, image, list_lt, list_bbox, list_labels)
    return

# -------------------------------------------------------
# cli

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inimg", required=True, help="directory containing JPG files. files must be at the root of the directory and end with `.jpg` extension to be processed")
    parser.add_argument("-b", "--inbbox", required=True, help="directory containing bounding box JSONS for each JPG file. files must be at the root and filenames must match the ones in `inimg`(minus the extension)")
    parser.add_argument("-o", "--output", required=True, help="output directory for the character detection. one file per input file will be saved")
    parser.add_argument("-v", "--visualize", action="store_true", default=False, help="visualize the character extraction results instead saving them. in this case, only the first 10 files will be processed.")
    parser.add_argument("-s", "--sample", action="store_true", default=False, help="process only the 10 first images.")
    args = parser.parse_args()

    inimg_dir = args.inimg
    inbbox_dir = args.inbbox
    output_dir = args.output
    visualize = args.visualize
    sample = args.sample

    if visualize:
        print("\nINFO: when using `-v` `--visualize` flag, at most 10 images are processed\n")
        sample = True

    inimg_dir, inbbox_dir, output_dir = sanitize(inimg_dir, inbbox_dir, output_dir)
    create_output_structure(inimg_dir, output_dir)
    file_pairs = get_file_pairs(inimg_dir, inbbox_dir, sample)

    outfiles = os.listdir(output_dir)
    outfiles = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(output_dir)
        for f in filenames if os.path.isfile(os.path.join(dp, f))
    ]

    charset, model, postprocessors = load_model()

    for (fp_json, fp_img) in tqdm(file_pairs, desc="processing character detection"):
        pipeline(model, postprocessors, charset, fp_json, fp_img, output_dir, visualize)


if __name__ == "__main__":
    cli()
