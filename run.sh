#!/bin/bash

DETR=advanced python main.py --backbone swin_large --drop-path-rate 0.5 --num-queries-one2one 900 --num-queries-one2many 1500 --pretrained checkpoints/h-defm-detr-swinL-dp0-mqs-lft-iter-2stg-hicodet.pth --output-dir outputs/alpha_0.2 --batch-size 6 --contact-alpha 0.2
DETR=advanced python main.py --backbone swin_large --drop-path-rate 0.5 --num-queries-one2one 900 --num-queries-one2many 1500 --pretrained checkpoints/h-defm-detr-swinL-dp0-mqs-lft-iter-2stg-hicodet.pth --output-dir outputs/alpha_0.8 --batch-size 6 --contact-alpha 0.8