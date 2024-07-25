"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import os
import time

import PIL.Image
import torch
import pickle
import numpy as np
import scipy.io as sio
from torchvision import transforms

from HOT.hot.lib.nn import async_copy_to
from HOT.hot.lib.utils import as_numpy

try:
    import wandb
except ImportError:
    pass

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

from ops import recover_boxes
from detr.datasets import transforms as T
from HOT.hot.utils import colorEncode

# HOT
import torch.nn as nn

from PIL import Image
# Our libs
import sys

sys.path.append('HOT')
from hot.config import cfg
from hot.dataset import TestDataset
from hot.models import ModelBuilder, SegmentationModule
from hot.utils import colorEncode
from hot.lib.nn import user_scattered_collate, async_copy_to
from hot.lib.utils import as_numpy

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = PIL.Image.NEAREST
    elif interp == 'bilinear':
        resample = PIL.Image.BILINEAR
    elif interp == 'bicubic':
        resample = PIL.Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])


def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    img = normalize(torch.from_numpy(img.copy()))
    return img


def process_img(image):
    """HOT处理图片"""
    img = image.convert('RGB')
    ori_width, ori_height = img.size
    img_resized_list = []
    imgSizes = (300, 375, 450, 525, 600)
    imgMaxSize = 400
    padding_constant = 8
    for this_short_size in imgSizes:
        # calculate target height and width
        scale = min(this_short_size / float(min(ori_height, ori_width)),
                    imgMaxSize / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        # to avoid rounding in network
        target_width = round2nearest_multiple(target_width, padding_constant)
        target_height = round2nearest_multiple(target_height, padding_constant)

        # resize images
        img_resized = imresize(img, (target_width, target_height), interp='bilinear')

        # image transform, to torch float tensor 3xHxW
        img_resized = img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)

    output = dict()
    output['img_ori'] = np.array(img)
    output['img_data'] = [x.contiguous() for x in img_resized_list]

    return output


with open('HOT/data/colors.npy', 'rb') as f:
    colors = np.load(f)


def hot_evaluate(hot_model, gpu, batch_data):
    hot_model.eval()

    segSize = (batch_data['img_ori'].shape[0],
               batch_data['img_ori'].shape[1])

    img_resized_list = batch_data['img_data']

    torch.cuda.synchronize()
    with torch.no_grad():
        scores = torch.zeros(1, 18, segSize[0], segSize[1])
        scores = async_copy_to(scores, gpu)

        # scores_part = torch.zeros(1, 18, segSize[0], segSize[1])
        # scores_part = async_copy_to(scores, gpu)

        for img in img_resized_list:
            feed_dict = batch_data.copy()
            feed_dict['img_data'] = img
            del feed_dict['img_ori']

            feed_dict = async_copy_to(feed_dict, gpu)

            # forward pass
            scores_tmp, scores_part_tmp = hot_model(feed_dict, segSize=segSize)
            scores = scores + scores_tmp / 5.0
            # scores_part = scores_part_tmp + scores_part_tmp / 5.0
            del feed_dict

        _, pred = torch.max(scores, dim=1)
        pred = pred.squeeze(0)

        # _, pred_part = torch.max(scores_part, dim=1)
        # pred_part = pred_part.squeeze(0)

    torch.cuda.synchronize()

    # visualization

    # pred_color = colorEncode(pred, colors)
    # pred_color_im = PIL.Image.fromarray(pred_color)
    # return transform_contact_img(pred_color_im)

    return pred.cpu()


def custom_collate(batch):
    images = []
    targets = []
    contact_masks = []
    # part_masks = []
    for im, mask, tar in batch:
        contact_masks.append(mask)
        # part_masks.append(p_mask)
        images.append(im)
        targets.append(tar)
    return images, contact_masks, targets


import os
import sys
import torch
import random
import warnings
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from caicl import build_detector
# from utils import custom_collate, CustomisedDLE, DataFactory
from configs import base_detector_args, advanced_detector_args

warnings.filterwarnings("ignore")


def get_cfg():
    hot_parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    hot_parser.add_argument(
        "--cfg",
        default="HOT/config/hot-resnet50dilated-c1.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    hot_parser.add_argument(
        "--epoch",
        default=14,
        help="which epoch"
    )
    hot_parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    cfg.merge_from_file("HOT/config/hot-resnet50dilated-c1.yaml")
    # cfg.freeze()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        # cfg.DIR, 'encoder_epoch_' + args.epoch + '.pth')
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        # cfg.DIR, 'decoder_epoch_' + args.epoch + '.pth')
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
           os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"
    if not os.path.isdir(os.path.join(cfg.DIR, "result_" + cfg.TEST.checkpoint.split('.')[0])):
        os.makedirs(os.path.join(cfg.DIR, "result_" + cfg.TEST.checkpoint.split('.')[0]))

    return cfg


# 获取配置参数
cfg = get_cfg()
# 建立HOT模型
net_encoder = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    weights=cfg.MODEL.weights_encoder)
net_decoder = ModelBuilder.build_decoder(
    cfg=cfg,
    arch=cfg.MODEL.arch_decoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    weights=cfg.MODEL.weights_decoder,
    use_softmax=True)

crit = nn.NLLLoss(ignore_index=-1)


hot_model = SegmentationModule(net_encoder, net_decoder, crit)
hot_model.to(0)

def main(rank, args):
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)

    trainset = DataFactory(
        name=args.dataset, partition=args.partitions[0],
        data_root=args.data_root
    )
    testset = DataFactory(
        name=args.dataset, partition=args.partitions[1],
        data_root=args.data_root
    )

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size // args.world_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            trainset, num_replicas=args.world_size,
            rank=rank, drop_last=True)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=args.batch_size // args.world_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            testset, num_replicas=args.world_size,
            rank=rank, drop_last=True)
    )

    if args.dataset == 'hicodet':
        object_to_target = train_loader.dataset.dataset.object_to_verb
        args.num_verbs = 117
    elif args.dataset == 'vcoco':
        object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        args.num_verbs = 24

    model = build_detector(args, object_to_target)

    if os.path.exists(args.resume):
        print(f"=> Rank {rank}: CAICL loaded from saved checkpoint {args.resume}.")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Rank {rank}: CAICL randomly initialised.")

    engine = CustomisedDLE(model, train_loader, test_loader, args)

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    if args.eval:
        if args.dataset == 'vcoco':
            """
            NOTE This evaluation results on V-COCO do not necessarily follow the 
            protocol as the official evaluation code, and so are only used for
            diagnostic purposes.
            """
            ap = engine.test_vcoco()
            if rank == 0:
                print(f"The mAP is {ap.mean():.4f}.")
            return
        else:
            ap = engine.test_hico()
            if rank == 0:
                # Fetch indices for rare and non-rare classes
                rare = trainset.dataset.rare
                non_rare = trainset.dataset.non_rare
                print(
                    f"The mAP is {ap.mean():.4f},"
                    f" rare: {ap[rare].mean():.4f},"
                    f" none-rare: {ap[non_rare].mean():.4f}"
                )
            return

    model.freeze_detector()
    param_dicts = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    optim = torch.optim.AdamW(param_dicts, lr=args.lr_head, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop, gamma=args.lr_drop_factor)
    # Override optimiser and learning rate scheduler
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    engine(args.epochs)


@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    args.num_verbs = 117
    args.num_triplets = 600
    object_to_target = dataset.dataset.object_to_verb
    model = build_detector(args, object_to_target)
    if args.eval:
        model.eval()
    if os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        print(f"Loading checkpoints from {args.resume}.")
        model.load_state_dict(ckpt['model_state_dict'])

    image, target = dataset[998]
    outputs = model([image], targets=[target])


if __name__ == '__main__':

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

    parser.add_argument('--port', default='1233', type=str)
    parser.add_argument('--seed', default=140, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')


    args = parser.parse_args()
    print(args)

    if args.sanity:
        sanity_check(args)
        sys.exit()
    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    mp.spawn(main, nprocs=args.world_size, args=(args,))


class DataFactory(Dataset):
    def __init__(self, name, partition, data_root):
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, "hico_20160224_det/images", partition),
                anno_file=os.path.join(data_root, f"instances_{partition}.json"),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, f"instances_vcoco_{partition}.json"),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )

            # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms_before = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
            ])
            self.transforms_after = T.Compose([
                T.ColorJitter(.4, .4, .4),
                normalize,
            ])
        else:
            self.transforms_before = T.Compose([
                T.RandomResize([800], max_size=1333),
            ])
            self.transforms_after = T.Compose([
                normalize,
            ])
        self.name = name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, target = self.dataset[i]
        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')

        target['image_verb_labels'] = torch.zeros(self.dataset.num_action_cls, dtype=torch.float32)
        if len(target['labels']) != 0:
            target['image_verb_labels'][target['labels']] = 1

        image, target = self.transforms_before(image, target)

        temp = process_img(image)

        contact_mask = hot_evaluate(hot_model, 0, temp)
        # contact_mask = torch.tensor(0)
        # target['contact_mask'] = contact_mask

        image, target = self.transforms_after(image, target)

        return image, contact_mask, target


class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v

    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]


class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, train_dataloader, test_dataloader, config):
        super().__init__(
            net, None, train_dataloader,
            print_interval=config.print_interval,
            cache_dir=config.output_dir,
            find_unused_parameters=True
        )
        self.config = config
        self.max_norm = config.clip_max_norm
        self.test_dataloader = test_dataloader

    def _on_start(self):
        if self._train_loader.dataset.name == "hicodet":
            # ap = self.test_hico()
            if self._rank == 0:
                # Fetch indices for rare and non-rare classes
                rare = self.test_dataloader.dataset.dataset.rare
                non_rare = self.test_dataloader.dataset.dataset.non_rare
                # perf = [ap.mean().item(), ap[rare].mean().item(), ap[non_rare].mean().item()]
                perf = [0, 0, 0]
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}, rare: {perf[1]:.4f}, none-rare: {perf[2]:.4f}."
                )
                self.best_perf = perf[0]
                wandb.init(config=self.config)
                wandb.watch(self._state.net.module)
                wandb.define_metric("epochs")
                wandb.define_metric("mAP full", step_metric="epochs", summary="max")
                wandb.define_metric("mAP rare", step_metric="epochs", summary="max")
                wandb.define_metric("mAP non_rare", step_metric="epochs", summary="max")

                wandb.define_metric("training_steps")
                wandb.define_metric("elapsed_time", step_metric="training_steps", summary="max")
                wandb.define_metric("loss", step_metric="training_steps", summary="min")

                wandb.log({
                    "epochs": self._state.epoch, "mAP full": perf[0],
                    "mAP rare": perf[1], "mAP non_rare": perf[2]
                })
        else:
            ap = self.test_vcoco()
            if self._rank == 0:
                perf = [ap.mean().item(), ]
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}."
                )
                self.best_perf = perf[0]
                """
                NOTE wandb was not setup for V-COCO as the dataset was only used for evaluation
                """
                wandb.init(config=self.config)

    # def _on_start(self):
    #     if self._train_loader.dataset.name == "hicodet":
    #         ap = self.test_hico()
    #         if self._rank == 0:
    #             # Fetch indices for rare and non-rare classes
    #             rare = self.test_dataloader.dataset.dataset.rare
    #             non_rare = self.test_dataloader.dataset.dataset.non_rare
    #             perf = [ap.mean().item(), ap[rare].mean().item(), ap[non_rare].mean().item()]
    #             # perf = [0, 0, 0]
    #             print(
    #                 f"Epoch {self._state.epoch} =>\t"
    #                 f"mAP: {perf[0]:.4f}, rare: {perf[1]:.4f}, none-rare: {perf[2]:.4f}."
    #             )
    #             self.best_perf = perf[0]
    #             wandb.init(config=self.config)
    #             wandb.watch(self._state.net.module)
    #             wandb.define_metric("epochs")
    #             wandb.define_metric("mAP full", step_metric="epochs", summary="max")
    #             wandb.define_metric("mAP rare", step_metric="epochs", summary="max")
    #             wandb.define_metric("mAP non_rare", step_metric="epochs", summary="max")
    #
    #             wandb.define_metric("training_steps")
    #             wandb.define_metric("elapsed_time", step_metric="training_steps", summary="max")
    #             wandb.define_metric("loss", step_metric="training_steps", summary="min")
    #
    #             wandb.log({
    #                 "epochs": self._state.epoch, "mAP full": perf[0],
    #                 "mAP rare": perf[1], "mAP non_rare": perf[2]
    #             })
    #     else:
    #         ap = self.test_vcoco()
    #         if self._rank == 0:
    #             perf = [ap.mean().item(), ]
    #             print(
    #                 f"Epoch {self._state.epoch} =>\t"
    #                 f"mAP: {perf[0]:.4f}."
    #             )
    #             self.best_perf = perf[0]
    #             """
    #             NOTE wandb was not setup for V-COCO as the dataset was only used for evaluation
    #             """
    #             wandb.init(config=self.config)

    def _on_end(self):
        if self._rank == 0:
            wandb.finish()

    def _on_each_iteration(self):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)


        if loss_dict['cls_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")


        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    def _print_statistics(self):
        running_loss = self._state.running_loss.mean()
        t_data = self._state.t_data.sum() / self._world_size
        t_iter = self._state.t_iteration.sum() / self._world_size

        # Print stats in the master process
        if self._rank == 0:
            num_iter = len(self._train_loader)
            n_d = len(str(num_iter))
            import sys
            _console_ = sys.stdout
            f = open('outs.txt', "a")
            sys.stdout = f
            print(
                "Epoch [{}/{}], Iter. [{}/{}], "
                "Loss: {:.4f}, "
                "Time[Data/Iter.]: [{:.2f}s/{:.2f}s]".format(
                    self._state.epoch, self.epochs,
                    str(self._state.iteration - num_iter * (self._state.epoch - 1)).zfill(n_d),
                    num_iter, running_loss, t_data, t_iter
                ))
            sys.stdout = _console_
            wandb.log({
                "elapsed_time": (time.time() - self._dawn) / 3600,
                "training_steps": self._state.iteration,
                "loss": running_loss
            })
        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def _on_end_epoch(self):
        if self._train_loader.dataset.name == "hicodet":
            ap = self.test_hico()
            if self._rank == 0:
                # Fetch indices for rare and non-rare classes
                rare = self.test_dataloader.dataset.dataset.rare
                non_rare = self.test_dataloader.dataset.dataset.non_rare
                perf = [ap.mean().item(), ap[rare].mean().item(), ap[non_rare].mean().item()]
                # ---------
                import sys
                _console_ = sys.stdout
                f = open('outs.txt', "a")
                sys.stdout = f
                # ---------
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}, rare: {perf[1]:.4f}, none-rare: {perf[2]:.4f}."
                )
                sys.stdout = _console_
                wandb.log({
                    "epochs": self._state.epoch, "mAP full": perf[0],
                    "mAP rare": perf[1], "mAP non_rare": perf[2]
                })
        else:
            ap = self.test_vcoco()
            if self._rank == 0:
                perf = [ap.mean().item(), ]
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}."
                )
                """
                NOTE wandb was not setup for V-COCO as the dataset was only used for evaluation
                """

        if self._rank == 0:
            # Save checkpoints
            checkpoint = {
                'iteration': self._state.iteration,
                'epoch': self._state.epoch,
                'performance': perf,
                'model_state_dict': self._state.net.module.state_dict(),
                'optim_state_dict': self._state.optimizer.state_dict(),
                'scaler_state_dict': self._state.scaler.state_dict()
            }
            if self._state.lr_scheduler is not None:
                checkpoint['scheduler_state_dict'] = self._state.lr_scheduler.state_dict()
            # torch.save(checkpoint, os.path.join(self._cache_dir, "latest.pth"))
            if perf[0] > self.best_perf:
                self.best_perf = perf[0]
                torch.save(checkpoint, os.path.join(self._cache_dir, "best.pth"))
            torch.save(checkpoint, os.path.join(self._cache_dir, str(self._state.epoch) + ".pth"))
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()

    @torch.no_grad()
    def test_hico(self):
        dataloader = self.test_dataloader
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))

        if self._rank == 0:
            meter = DetectionAPMeter(
                600, nproc=1, algorithm='11P',
                num_gt=dataset.anno_interaction,
            )
        for batch in tqdm(dataloader, disable=(self._world_size != 1)):
            # mark有问题
            inputs = pocket.ops.relocate_to_cuda(batch[:-1])
            outputs = net(*inputs)
            outputs = pocket.ops.relocate_to_cpu(outputs, ignore=True)
            targets = batch[-1]
            # contact_masks = batch[-1]

            scores_clt = [];
            preds_clt = [];
            labels_clt = []
            for output, target in zip(outputs, targets):
                # Format detections
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
                scores = output['scores']
                verbs = output['labels']
                objects = output['objects']
                interactions = conversion[objects, verbs]
                # Recover target box scale
                gt_bx_h = recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = recover_boxes(target['boxes_o'], target['size'])

                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)
                unique_hoi = interactions.unique()
                for hoi_idx in unique_hoi:
                    gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                    det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                             gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                             boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )

                scores_clt.append(scores)
                preds_clt.append(interactions)
                labels_clt.append(labels)
            # Collate results into one tensor
            scores_clt = torch.cat(scores_clt)
            preds_clt = torch.cat(preds_clt)
            labels_clt = torch.cat(labels_clt)
            # Gather data from all processes
            scores_ddp = pocket.utils.all_gather(scores_clt)
            preds_ddp = pocket.utils.all_gather(preds_clt)
            labels_ddp = pocket.utils.all_gather(labels_clt)

            if self._rank == 0:
                meter.append(torch.cat(scores_ddp), torch.cat(preds_ddp), torch.cat(labels_ddp))

        if self._rank == 0:
            ap = meter.eval()
            return ap
        else:
            return -1

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, (image, target) in enumerate(tqdm(dataloader.dataset)):
            inputs = pocket.ops.relocate_to_cuda([image, ])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num

        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def test_vcoco(self):
        dataloader = self.test_dataloader
        net = self._state.net;
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)

        if self._rank == 0:
            meter = DetectionAPMeter(
                24, nproc=1, algorithm='11P',
                num_gt=dataset.num_instances,
            )
        for batch in tqdm(dataloader, disable=(self._world_size != 1)):
            inputs = pocket.ops.relocate_to_cuda(batch[:-1])
            outputs = net(*inputs)
            outputs = pocket.ops.relocate_to_cpu(outputs, ignore=True)
            targets = batch[-1]

            scores_clt = [];
            preds_clt = [];
            labels_clt = []
            for output, target in zip(outputs, targets):
                # Format detections
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
                scores = output['scores']
                actions = output['labels']
                gt_bx_h = recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = recover_boxes(target['boxes_o'], target['size'])

                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)
                unique_actions = actions.unique()
                for act_idx in unique_actions:
                    gt_idx = torch.nonzero(target['actions'] == act_idx).squeeze(1)
                    det_idx = torch.nonzero(actions == act_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                             gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                             boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )

                scores_clt.append(scores)
                preds_clt.append(actions)
                labels_clt.append(labels)
            # Collate results into one tensor
            scores_clt = torch.cat(scores_clt)
            preds_clt = torch.cat(preds_clt)
            labels_clt = torch.cat(labels_clt)
            # Gather data from all processes
            scores_ddp = pocket.utils.all_gather(scores_clt)
            preds_ddp = pocket.utils.all_gather(preds_clt)
            labels_ddp = pocket.utils.all_gather(labels_clt)

            if self._rank == 0:
                meter.append(torch.cat(scores_ddp), torch.cat(preds_ddp), torch.cat(labels_ddp))

        if self._rank == 0:
            ap = meter.eval()
            return ap
        else:
            return -1

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, (image, target) in enumerate(tqdm(dataloader.dataset)):
            inputs = pocket.ops.relocate_to_cuda([image, ])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)
