import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
# import pandas as pd

# np.random.seed([3, 1415])
# n = 20
# cols = np.array[1,2,3,4,5]
# arr1 = (np.random.randint(5, size=(n,4)))
# # img1 = mpimg.imread('boothiGray.jpg')
# # img2 = mpimg.imread('groupGray.jpg')
# # li2 = [[1,2],[3,4],[5,6]]
# # li2 = np.asarray(li2)



from PIL import Image

# im1 = Image.open(r'testboothi.png')
# im2 = im1.convert('RGB')
# im2.save(r'testboothi8.jpg')
i2 = mpimg.imread('lidd.jpg')
i2 = i2[:, :, 2]
print(i2.shape)


# # # li2[1]= 1
# # # print(li2)
# # li4 = li2.copy()
# # print(li2.shape)
# # for r in range(li2.shape[0]):
# #     for c in range(li2.shape[1]):
# #         li2[c][r] = li4[r][c]
# # print(li4)
# # print(li2)
# # # li2[2]= 3
# # # print(li2)
# # # print(li4)
# # # li4[2]=5
# # # print(li2)
# # # print(li4)
# # # temp_img_1 = img1
# # # print(temp_img_1)
# # # print(img1)
# # # temp_img_1[0][0]==0
# # # print("-------------------------------------")
# # # print(temp_img_1)
# # # print(img1)
# # # temp_img_2 = img2
# # # print()
# # # print(img1.shape[0])
# # # img1.shape[0]=temp_1.shape[1]
# # # img1.shape[1]=temp_1.shape[0]
# # # img2.shape[0]=temp_2.shape[1]
# # # img2.shape[1]=temp_2.shape[0]
# # # print(img2[0][0])
# # # print(img2[1])
# # # for r in img2.shape[0]:
# # #     for c in img2.shape[1]:
   



 # # NUMERATOR
        # a = frames[i]-np.mean(frames[i])
        # b = img1-np.mean(img1)
        # if (a.shape==b.shape):
        #     mul = a*b    
        #     mul_sum = np.sum(mul)
        # # dENOMINATOR
        #     a_sq = a**2
        #     b_sq = b**2
        #     a_sum = np.sum(a_sq)
        #     b_sum = np.sum(b_sq)
        #     result= np.sqrt(a_sum*b_sum)
        # # CROSS-CORRELATION
        #     value = mul_sum/result
              # Normalise X and Y