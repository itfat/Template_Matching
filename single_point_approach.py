import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np

i1 = mpimg.imread('faceGray.jpg')


i2 = mpimg.imread('groupGray.jpg')
# FOR PNG
# i1 = i1[:,:,2]
img1 = i1.copy()
img2 = i2.copy()

img1 = img1.T
img2 = img2.T
pop_initial =(np.random.randint(img2.shape[0]), np.random.randint(img2.shape[1]))
frame = img2[pop_initial[0]:(pop_initial[0]+img1.shape[0]),pop_initial[1]:(pop_initial[1]+img1.shape[1])]
# Baba Boothi Correlation
if frame.shape == img1.shape:
    frame = frame - frame.mean()
    img1 = img1 - img1.mean()
    frame = frame / frame.std()
    img1 = img1 / img1.std()
    value = np.mean(frame*img1)
else:
    value = 0
print(value)
if value >= 0.2:
    fig,ax = plt.subplots(1)
    ax.imshow(i2, cmap= 'gray')
    rect = patches.Rectangle(pop_initial,29,35,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.xticks(rotation=90)
    plt.show()
# else:


# sorted_fitness = []
# # # print(count)
# # # print(frames)
# fitness_values = []
# for i in range(count):
#     if frames[i].shape == img1.shape:
#         frames[i] = frames[i] - frames[i].mean()
#         img1 = img1 - img1.mean()
#         frames[i] = frames[i] / frames[i].std()
#         img1 = img1 / img1.std()
#         value = np.mean(frames[i]*img1)
#         fitness_values.append(value)
#     else:
#         fitness_values.append(0)

# sorted_fitness = sorted(fitness_values)
# print(sorted_fitness)
# print(coordinates)

# for i in range(count):
#     if sorted_fitness[count-1] == fitness_values[i]:
#         max_point = coordinates[i]



# # GRAPH TO SEE THE RELATION
# # fitness_values = np.asarray(fitness_values)
# print(generations)
# plt.plot(generations, fitness_values, label = 'max')
# plt.title('Fitness against each generation')
# plt.xlabel('No. of Generations')
# plt.ylabel('Max Fitness')
# plt.show()

