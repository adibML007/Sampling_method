import nrrd
import numpy as np
from PIL import Image
box = (100, 100, 438, 438)
# For lgemri traning data
# for i in range(16):
#     filepath1 = f'E:/Training_data/lgemri{i}.nrrd'
#     filepath = f'E:/Training_data/laendo{i}.nrrd'
#     img = nrrd.read(filepath)
#     img1 = nrrd.read(filepath1)
#     for j in range(44):
#         temp = img[0][:, :, j]
#         if temp.sum() > 0
#             im1 = Image.fromarray(np.uint8(img1[0][:, :, j]))
#             im1 = im1.crop(box)
#             im1 = im1.convert("RGB")
#             im1.save(f"train_hq/pt{i}_mri{j}.jpg")

# For laendo mask data
for i in range(16):
    filepath = f'E:/Training_data/laendo{i}.nrrd'
    img = nrrd.read(filepath)
    for j in range(44):
        temp = img[0][:, :, j]
        if temp.sum() > 0:
            im1 = Image.fromarray(np.uint8(img[0][:, :, j])*255)
            im1 = im1.crop(box)
            im1.save(f"C:/Users/Adib/PycharmProjects/Pytorch-UNet-master/data/masks/pt{i}_mri{j}_mask.gif")



# filepath = 'E:/Training_data/laendo0.nrrd'
# img = nrrd.read(filepath)
# im1 = Image.fromarray(np.uint8(img[0][:, :, 27]), 'L')
# print(np.unique(im1))
# im1 = im1.crop(box)
# print(np.unique(im1))
# im1.save("output.png")
# T = np.asarray(Image.open('output.png'))
# print(np.unique(list(T)))
