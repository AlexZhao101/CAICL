遵循https://github.com/fredzzhang/pvic中的步骤安装环境，下载数据库并创建软链接

下载H-DETR权重放在checkpoints中：
https://drive.google.com/file/d/1wge-CC1Fx67EHOSXyHGHvrqvMva2jEkr/view?usp=share_link

下载HOT权重放在HOT/ckpt/hot-c1中：
https://download.is.tue.mpg.de/download.php?domain=hot&resume=1&sfile=hot-c1.zip


#训练
‘’‘
--backbone
swin_large
--drop-path-rate
0.5
--num-queries-one2one
900
--num-queries-one2many
1500
--pretrained
checkpoints/h-defm-detr-swinL-dp0-mqs-lft-iter-2stg-hicodet.pth
--output-dir
outputs/test777
--batch-size
4
--contactalpha
0.2
--human-scale
38
’‘’
