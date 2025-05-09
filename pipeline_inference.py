import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch import nn
from PIL import Image

import datasets.transforms as T
from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

# -------------------------------------------------------
# paths

CWD_PATH = os.getcwd()  # directory from which the cli is run
DIR_PATH = os.path.abspath(os.path.dirname(__file__))  # directory of this file
MODEL_CONFIG_PATH = os.path.join(DIR_PATH, "config", "HWDB_full.py")
MODEL_CHARSET_PATH = os.path.join(DIR_PATH, "data", "dante", "labels_icdar.pkl")
MODEL_CHECKPOINT_PATH = os.path.join(DIR_PATH, "logs", "dante", "medieval_checkpoint.pth")

# WORKSSSSSS
with open(MODEL_CHARSET_PATH, mode="rb") as fh:
    charset = pickle.load(fh)

# -------------------------------------------------------
# model

#TODO understand
args = SLConfig.fromfile(MODEL_CONFIG_PATH)
args.device = 'cuda:0'
args.CTC_training = False
args.CTC_loss_coef = 0.25
args.dataset_file = 'icdar_multi'

args.coco_path = "/comp_robot/cv_public_dataset/COCO2017/"  # the path of coco
args.fix_size = False

# I THINK WE SHOULD MODIFY THIS WITH `labels_idcar.py`
# args.dataset_file = "icdar_classif_font" #icdar_" + args.dataset_file
dataset_val = build_dataset(image_set='train', args=args)

device = args.device
args.charset = dataset_val.charset
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location='cpu')

device = args.device
args.charset = dataset_val.charset
model, criterion, postprocessors = build_model_main(args)
features_dim = model.class_embed[0].weight.data.shape[1]
new_charset_size = len(args.charset)
new_class_embed = nn.Linear(features_dim, new_charset_size, )
new_decoder_class_embed = nn.Linear(features_dim, new_charset_size, )
new_enc_out_class_embed = nn.Linear(features_dim, new_charset_size, )

if model.dec_pred_class_embed_share:
    class_embed_layerlist = [new_class_embed for i in range(model.transformer.num_decoder_layers)]

### Injective mapping between the new and old charset (random mapping)

# charset_without_accent = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
#                       't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
#                       'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4',
#                       '5', '6', '7', '8', '9', '!', '?', ]
# symbols = ['"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '@', '[',
#        '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ']
# accent_charset = ['à', 'á', 'â', 'ã', 'ä', 'å', 'ā', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò',
#               'ó', 'ô', 'õ', 'ö', 'ō', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ', 'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ',
#               'Ç', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í', 'Î', 'Ï', 'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', 'Ø', 'Ù', 'Ú', 'Û',
#               'Ü', 'Ý', 'Þ', 'Ÿ']
# weird_charset = ['«', '»', '—', "’", "°", "–", "œ"]
# old_charset = charset_without_accent + accent_charset + weird_charset + symbols
#
# not_mapped = []
# possible_mapping = list(range(len(old_charset)))
# mapping = {}
# for i, char in enumerate(args.charset):
#     if char in old_charset:
#         mapping[i] = old_charset.index(char)
#         possible_mapping.remove(mapping[i])
#     else:
#         not_mapped.append(char)
# print(len(mapping), len(not_mapped))
# while len(possible_mapping) < len(not_mapped):
#     possible_mapping.append(np.random.randint(0, len(old_charset)))
# possible_mapping = list(np.random.permutation(possible_mapping))
#
# for i, char in enumerate(args.charset):
#     if char not in old_charset:
#         mapping[i] = possible_mapping[0]
#         possible_mapping.pop(0)
#
# # check if all is mapped
# print(len(mapping), len(args.charset))
#
new_class_embed = nn.ModuleList(class_embed_layerlist)
# for j in range(model.transformer.num_decoder_layers):
#     for i in range(new_charset_size):
#         new_class_embed[j].weight.data[i, :] = model.class_embed[j].weight.data[mapping[i], :]
#         new_class_embed[j].bias.data[i] = model.class_embed[j].bias.data[mapping[i]]
#
#         new_decoder_class_embed.weight.data[i, :] = model.transformer.decoder.class_embed[j].weight.data[mapping[i], :]
#         new_decoder_class_embed.bias.data[i] = model.transformer.decoder.class_embed[j].bias.data[mapping[i]]
#
#         new_enc_out_class_embed.weight.data[i, :] = model.transformer.enc_out_class_embed.weight.data[mapping[i], :]
#         new_enc_out_class_embed.bias.data[i] = model.transformer.enc_out_class_embed.bias.data[mapping[i]]

model.class_embed = new_class_embed.to(device)
model.transformer.decoder.class_embed = new_decoder_class_embed.to(device)
model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)
# model.transformer.enc_out_class_embed = new_enc_out_class_embed.to(device)

# if model.label_enc.weight.data.shape[0] < len(dataset_val.charset)+1:
model.label_enc = nn.Embedding(len(dataset_val.charset) + 1, features_dim).to(device)
# checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)

# -------------------------------------------------------
# load data

# images
folder_data = '/home/rbaena/Downloads/dantes/'
# bounding boxes
path_annotations = '/home/rbaena/projects/OCR/OCR_line/DINO/DANTES_annos'

# dict_book = { "witness/folder": ["list of path to images"] }
list_book = os.listdir(folder_data)
dict_book = {}
for book in list_book:
    list_pages = os.listdir(folder_data + book)
    dict_book[book] = list_pages
    # sort
    dict_book[book].sort()

# all pages to be processed in total
total_number_image = 0
for bb in dict_book:
    total_number_image += len(dict_book[bb])

# -------------------------------------------------------
# pipeline

it = 0
# dict_book lien avec des path => bb = pages
for bb in dict_book:
    for pp in dict_book[bb]:
        idx = pp.split('.')[0]

        # WHOLE IMAGE
        path_im = folder_data + bb + '/' + pp

        image = Image.open(path_im).convert("RGB")

        path_anno = path_annotations + '/' + bb + '/' + idx + '.json'
        with open(path_anno) as f:
            # bounding boxes extracted from final poly
            data = json.load(f)
        list_bbox = []
        list_left_top = []
        list_labels = []
        for dd in data['annotations']:
            bbox_crop = dd['bbox']
            # crop image
            left, top, right, bottom = bbox_crop
            # widen bounding boxes
            left = left - 10
            right = right + 10
            top = top - 10
            bottom = bottom + 3
            try:
                list_left_top.append((left, top))
                crop = image.crop((left, top, right, bottom))

                # # creqte folder crop for save
                # if not os.path.exists('crops'):
                #     os.makedirs('crops')
                # crop.save('crops/'+pp.split('.')[0]+'_'+str(it)+'.jpg')

                transform = T.Compose([
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                orig_size = crop.size
                crop_image, _ = transform(crop, None)
                actual_size = crop_image.shape[2], crop_image.shape[1]
            except Exception as e:
                print(e)
                print('Error in page', pp)
                continue
            try:
                with torch.no_grad():
                    output = model.cuda()(crop_image[None].cuda())
                    polygones = output['pred_boxes']

                    postprocessors['bbox'].nms_iou_threshold = 0.2
                    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

                    boxes = output['boxes']
                    scores = output['scores']
                    labels = output['labels']
                    select_mask = scores > 0.1
                    boxes_xyxy = boxes.clone()
                    boxes = box_ops.box_xyxy_to_cxcywh(boxes)
                    boxes = boxes[select_mask]
                    scores = scores[select_mask]
                    labels = labels[select_mask]
                    box_label = [dataset_val.charset[i] for i in labels]
                    # TODO retrieve labels corresponding to " " => delete to not consider accents
                    box_label = [bytes(_string, "utf-8").decode("unicode_escape") for _string in box_label]
                    list_labels.append(box_label)

                ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(orig_size, actual_size))
                ratios_h, ratios_w = ratios[0], ratios[1]
                w, h = actual_size
                final_bboxes = boxes.cpu() * torch.tensor([w, h, w, h])  #
                final_bboxes[:, :2] -= final_bboxes[:, 2:] / 2
                final_bboxes *= torch.Tensor([ratios_w, ratios_h, ratios_w, ratios_h]) # Bboxes correspondant à la vraie image croppée
                list_bbox.append(final_bboxes)
            except Exception as e:
                print(e)
                print('Error in page', pp)
                list_bbox.append([])
                list_labels.append([])

        # create folder bb for
        if not os.path.exists('dantes_images'):
            os.makedirs('dantes_images')
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        bbox_image = []
        label_image = []
        for left_top, bbox, labels in zip(list_left_top, list_bbox, list_labels):
            for bbb in bbox:
                x, y, w, h = bbb
                x_shifted = x + left_top[0]
                y_shifted = y + left_top[1]
                rect = patches.Rectangle((x_shifted, y_shifted), w, h, linewidth=0.11, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # Shift des bboxes pour avoir position relatives dans images
                bbox_image.append(torch.tensor([x_shifted, y_shifted, w, h]).long())
            label_image.append(labels)
        try:
            bbox_image = torch.stack(bbox_image)
        except Exception as e:
            print(e)
            print('Error in page', pp)
            continue

        # flatten labels
        label_image = [item for sublist in label_image for item in sublist]
        if not os.path.exists('dantes_images/' + bb):
            os.makedirs('dantes_images/' + bb)

        plt.savefig('dantes_images/' + bb + '/' + pp, dpi=300)
        if not os.path.exists('dantes_coco'):
            os.makedirs('dantes_coco')

        # convert to coco format and save json
        coco_format = {
            'images': [{'file_name': pp, 'id': 0, 'height': image.size[1], 'width': image.size[0]}],
            'annotations': [],
            'categories': []
        }

        # To keep track of category IDs
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
        if not os.path.exists('dantes_coco/' + bb):
            os.makedirs('dantes_coco/' + bb)
        with open('dantes_coco/' + bb + '/' + pp.split('.')[0] + '.json', 'w') as f:
            json.dump(coco_format, f)

        print("\r Progress: {}/{}, done page {}".format(it, total_number_image, pp), end="")


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

# filenames have the structure: wit<witId>_man<manId>_<pageNum>.<extensiom>
# => extract wit<witId>_man<manId>
def wid_from_filename(f:str) -> str|None:
    wid = re.search(r"^wit\d+_man\d+", f)
    return wid[0] if wid is not None else None

# -------------------------------------------------------
# cli input functions

def sanitize(inimg_dir:str, inbbox_dir:str, outimg_dir:str) -> Tuple[os.PathLike]:
    inimg_dir_abs = to_abspath(inimg_dir)
    inbbox_dir_abs = to_abspath(inbbox_dir)
    outimg_dir_abs = to_abspath(outimg_dir)

    if not os.path.isdir(inimg_dir_abs):
        raise FileNotFoundError(f"input image directory '{inimg_dir}' not found. exiting (absolute path: '{inimg_dir_abs}')")
    if not os.path.isdir(inbbox_dir_abs):
        raise FileNotFoundError(f"input bounding box directory '{inbbox_dir_abs}' not found. exiting (absolute path: '{inbbox_dir_abs}')")

    # check that we have matching images for all bounding box files in inbbox_dir
    assert all(
        rmext(inbbox)
        in [rmext(inimg) for inimg in inimg_dir_abs]
        for inbbox in inbbox_dir_abs
    ), f"not all bounding boxes (in '{inbbox_dir_abs}') have a matching image file (in '{inimg_dir_abs}')"

    return inimg_dir_abs, inbbox_dir_abs, outimg_dir_abs

# create output directory: output_img_dir/<wid>/ (in `output_img_dir`, one directotry per `wid`)
def create_output_structure(inimg_dir:os.PathLike, outimg_dir:os.PathLike) -> None:
    if not os.path.isdir(outimg_dir):
        os.makedirs(outimg_dir_abs)
    witnesses = []
    for f in os.listdir(dir):
        wid = wid_from_filename(f)
        if wid is not None and wid not in witnesses:
            witnesses.append(wid)
    for wid in set(witneses):
        wid_path = os.path.join("outimg_dir_abs", wid)
        if not os.path.isdir(wid_path):
            os.makedirs(wid_path)
    return


# -------------------------------------------------------
# cli

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inimg", required=True, help="directory containing JPG files. files must be at the root of the directory and end with `.jpg` extension to be processed")
    parser.add_argument("-b", "--inbbox", required=True, help="directory containing bounding box jsons for each JPG file. files must be at the root and filenames must match the ones in `inimg`(minus the extension)")
    parser.add_argument("-o", "--output", required=True, help="output directory for the character detection. one file per input file will be saved")
    args = parser.parse_args()

    inimg_dir = args.inimg
    inbbox_dir = args.inbbox
    output_dir = args.output

    inimg_dir, inbbox_dir, output_dir = sanitize(inimg_dir, inbbox_dir, output_dir)
    create_output_structure(inimg_dir, output_dir)


if __name__ == "__main__":
    cli()
