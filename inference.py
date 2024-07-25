"""
Visualise detected human-object interactions in an image

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""
import json
import os
import math

import PIL
import torch
from torchvision.transforms import transforms

import pocket
import pocket.advis
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as peff
import torch.multiprocessing as mp
from mpl_toolkits.axes_grid1 import make_axes_locatable

from HOT.hot.lib.nn import async_copy_to
from configs import advanced_detector_args, base_detector_args
from main import DataFactory, round2nearest_multiple, get_cfg
from caicl import build_detector

from HOT.hot.utils import colorEncode

# HOT
import torch.nn as nn

from PIL import Image
# Our libs
import sys


warnings.filterwarnings("ignore")

OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def draw_boxes(ax, boxes):
    xy = boxes[:, :2].unbind(0)
    h, w = (boxes[:, 2:] - boxes[:, :2]).unbind(1)
    for i, (a, b, c) in enumerate(zip(xy, h.tolist(), w.tolist())):
        patch = patches.Rectangle(a.tolist(), b, c, facecolor='none', edgecolor='w')
        ax.add_patch(patch)
        txt = plt.text(*a.tolist(), str(i + 1), fontsize=20, fontweight='semibold', color='w')
        txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
        plt.draw()


def visualise_entire_image(image, output, attn, actions, action=None, thresh=0.2):
    """Visualise bounding box pairs in the whole image by classes"""
    # Rescale the boxes to original image size
    ow, oh = image.size
    h, w = output['size']
    scale_fct = torch.as_tensor([
        ow / w, oh / h, ow / w, oh / h
    ]).unsqueeze(0)
    boxes = output['boxes'] * scale_fct.cuda()

    image_copy = image.copy()
    scores = output['scores']
    # objects = output['objects']
    pred = output['labels']
    # Visualise detected human-object pairs with attached scores
    if action is not None:
        keep = torch.nonzero(torch.logical_and(scores >= thresh, pred == action)).squeeze(1)
        # keep = torch.nonzero(scores >= thresh).squeeze(1)

        bx_h, bx_o = boxes[output['pairing']].unbind(1)
        pocket.utils.draw_box_pairs(image, bx_h[keep].cpu(), bx_o[keep].cpu(), width=5)
        plt.imshow(image)
        plt.axis('off')

        for i in range(len(keep)):
            txt = plt.text(*bx_h[keep[i], :2], f"{scores[keep[i]]:.2f}", fontsize=15, fontweight='semibold', color='w')
            txt.set_path_effects([peff.withStroke(linewidth=5, foreground='#000000')])
            plt.draw()

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig("fig.png", bbox_inches="tight", pad_inches=0)

        for i in keep:
            ho_pair_idx = output["x"][i]
            attn_map = attn[0, :, ho_pair_idx].reshape(8, math.ceil(h / 32), math.ceil(w / 32))
            attn_image = image_copy.copy()
            pocket.utils.draw_boxes(attn_image, torch.stack([bx_h[i].cpu(), bx_o[i].cpu()]), width=4)
            for j in range(8):
                pocket.advis.heatmap(attn_image, attn_map[j: j+1], save_path=f"pair_{i}_attn_head_{j+1}.png")
                plt.close()

            # pocket.advis.heatmap(attn_image, attn_map[0: 9], save_path=f"add_heatmap.png")
            # plt.close()

def get_img_index(filename):
    empty = [153, 165, 221, 374, 410, 663, 702, 779, 1099, 1376, 1460, 1472, 1584, 1614, 1677, 1778, 1811, 1824, 1830,
              2002, 2058, 2106, 2180, 2255, 2268, 2363, 2366, 2509, 2549, 2557, 2944, 2970, 3050, 3130, 3173, 3196,
              3223, 3638, 3692, 3784, 3811, 3818, 4028, 4031, 4231, 4290, 4326, 4366, 4500, 4555, 4581, 4661, 4698,
              4806, 4809, 4894, 5043, 5161, 5194, 5357, 5573, 5806, 5819, 5835, 5865, 5908, 6036, 6197, 6503, 6639,
              6643, 6650, 6664, 6711, 6937, 6943, 7114, 7342, 7448, 7609, 7796, 7819, 7875, 8017, 8041, 8073, 8278,
              8325, 8406, 8427, 8428, 8620, 8973, 9091, 9218, 9219, 9226, 9232, 9254, 9266, 9269, 9350, 9351, 9361,
              9409, 9410, 9411, 9421, 9499, 9606, 9607, 9608]
    with open('hicodet/instances_test2015.json', 'r') as f:
        anno = json.load(f)
    i = anno['filenames'].index(filename)
    temp = i
    for ep in empty:
        if i > ep:
            temp -= 1
        else:
            break
    return temp

def hoi_list():
    hoi_dict={}
    with open("hico_list_hoi.txt", "r", encoding='utf-8') as f:  #打开文本
        for line in f.readlines():  # 依次读取每行
            line = line.strip().split()  # 去掉每行头尾空白
            hoi_dict.setdefault(int(line[0])-1,[]).append(line[2]+' '+line[1])
    return hoi_dict

def action_list():
    action_dict={}
    with open("hico_list_vb.txt", "r", encoding='utf-8') as f:  #打开文本
        for line in f.readlines():  # 依次读取每行
            line = line.strip().split()  # 去掉每行头尾空白
            action_dict.setdefault(line[1]).append(int(line[0])-1)
    return action_dict

def get_img_action(filename):
    with open('hicodet/instances_test2015.json', 'r') as f:
        anno = json.load(f)
    i = anno['filenames'].index(filename)
    hoi_infos = anno['annotation'][i]['hoi']
    hoi_dict = hoi_list()  # key是数字 value是['feed', 'zebra']
    vb_list = action_list()  # key是str value是list
    hoi_strs = []
    for idx in hoi_infos:
        hoi_strs.append(hoi_dict.get(idx)[0])
    vb_result = []
    for hoi_str in hoi_strs:
        tp = str(vb_list.get(hoi_str.split()[0])[0]) + " " + hoi_str
        vb_result.append(tp)
    return vb_result

@torch.no_grad()
def main(args):
    testset = DataFactory(
        name=args.dataset, partition=args.partitions[1],
        data_root=args.data_root
    )
    conversion = testset.dataset.object_to_verb if args.dataset == 'hicodet' \
        else list(testset.dataset.object_to_action.values())
    args.num_verbs = 117 if args.dataset == 'hicodet' else 24
    actions = testset.dataset.verbs if args.dataset == 'hicodet' else \
        testset.dataset.actions

    upt = build_detector(args, conversion)
    upt.eval()
    upt.cuda()
    attn_weights = []
    hook = upt.decoder.layers[-1].qk_attn.register_forward_hook(
        lambda self, input, output: attn_weights.append(output[1])
    )
    # hook2 = upt.decoder_mask.layers[-1].qk_attn.register_forward_hook(
    #     lambda self, input, output: attn_weights.append(output[1])
    # )
    if os.path.exists(args.resume):
        print(f"=> Continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        upt.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print(f"=> Start from a randomly initialised model")

    if args.image_path is None:
        image, contact_mask, _ = testset[args.index]
        output = upt([image.cuda()], [contact_mask.cuda()])
        image = testset.dataset.load_image(
            os.path.join(testset.dataset._root,
                         testset.dataset.filename(args.index)
                         ))
    else:
        image = testset.dataset.load_image(args.image_path)
        image_tensor, _ = testset.transforms_before(image, None)
        image_tensor, _ = testset.transforms_after(image, None)
        output = upt([image_tensor.cuda()])

    hook.remove()
    # hook2.remove()
    visualise_entire_image(
        # image, output[0], attn_weights[0].cpu() + attn_weights[1].cpu(),
        image, output[0], attn_weights[0].cpu(),
        actions, args.action, 0.05
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    if "DETR" not in os.environ:
        raise KeyError(f"Specify the detector type with env. variable \"DETR\".")
    elif os.environ["DETR"] == "base":
        parser = argparse.ArgumentParser(parents=[base_detector_args(), ])
        parser.add_argument('--detector', default='base', type=str)
        parser.add_argument('--raw-lambda', default=2.8, type=float)
    elif os.environ["DETR"] == "advanced":
        parser = argparse.ArgumentParser(parents=[advanced_detector_args(), ])
        parser.add_argument('--detector', default='advanced', type=str)
        parser.add_argument('--raw-lambda', default=1.7, type=float)

    parser.add_argument('--kv-src', default='C5', type=str, choices=['C5', 'C4', 'C3'])
    parser.add_argument('--repr-dim', default=384, type=int)
    parser.add_argument('--triplet-enc-layers', default=1, type=int)
    parser.add_argument('--triplet-dec-layers', default=2, type=int)

    parser.add_argument('--alpha', default=.5, type=float)
    parser.add_argument('--gamma', default=.1, type=float)
    parser.add_argument('--box-score-thresh', default=.05, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)
    parser.add_argument('--contactalpha', default=.2, type=float)
    parser.add_argument('--human-scale', default=40, type=int)
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--use-wandb', default=False, action='store_true')
    parser.add_argument('--index', default=7890, type=int)
    parser.add_argument('--port', default='1233', type=str)
    parser.add_argument('--seed', default=140, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--action', default=1, type=int)
    parser.add_argument('--image-path', default=None, type=str)
    args = parser.parse_args()

    main(args)

    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    # mp.spawn(main, nprocs=args.world_size, args=(args,))