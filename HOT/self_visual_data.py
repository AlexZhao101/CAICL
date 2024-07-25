# 将分割图和原图合在一起
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mp

#image1 原图 
#image2 分割图
# image1 = Image.open(r"E:\论文\HOT-Annotated\images\hake_train2015_HICO_train2015_00000001.jpg")
# image2 = Image.open(r"E:\论文\HOT-Annotated\segments\hake_train2015_HICO_train2015_00000001.png")
#
# image1 = image1.convert('RGBA')
# image2 = image2.convert('RGBA')
 
#两幅图像进行合并时，按公式：blended_img = img1 * (1 – alpha) + img2* alpha 进行
# image = Image.blend(image1,image2,0.)
# image.save("test.png")


seg = mp.imread(r"E:\dataset\HOT-Annotated\segments\hake_train2015_HICO_train2015_00000001.png")
anno = mp.imread(r"E:\dataset\HOT-Annotated\annotations\hake_train2015_HICO_train2015_00000001.png")
# img = mp.imread(r"E:\dataset\HOT-Annotated\images\hake_train2015_HICO_train2015_00000001.png")

# plt.figure()
# plt.imshow(img)

# plt.subplot(1,2,2)
# plt.imshow(seg)
plt.imshow(anno)
plt.show()



