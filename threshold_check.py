import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np

i1 = mpimg.imread('faceGray.jpg')

# i1 = mpimg.imread('testface.png')
i2 = mpimg.imread('groupGray.jpg')
# FOR PNG
# i1 = i1[:,:,2]
img1 = i1.copy()
img2 = i2.copy()

img1 = img1.T
img2 = img2.T
frames = []

coordinates = []
print(img1.shape)
count = 0
generations = []
for i in range(0,img2.shape[0], img1.shape[0]):

    for j in range(0,img2.shape[1], img1.shape[1]):
        frame = img2[i:(i+img1.shape[0]),j:(j+img1.shape[1])]
        frames.append(frame)
        coordinates.append((i,j))
        generations.append(i)
        count +=1
print(count)
# frame = img2[111:(111+img1.shape[0]),209:(209+img1.shape[1])]
# # Baba Boothi Correlation
# frame = frame - frame.mean()
# img1 = img1 - img1.mean()
# frame = frame / frame.std()
# img1 = img1 / img1.std()
# value = np.mean(frame*img1)
# print(value)
sorted_fitness = []
# # print(count)
# # print(frames)
fitness_values = []
for i in range(count):
    if frames[i].shape == img1.shape:
        frames[i] = frames[i] - frames[i].mean()
        img1 = img1 - img1.mean()
        frames[i] = frames[i] / frames[i].std()
        img1 = img1 / img1.std()
        value = np.mean(frames[i]*img1)
        fitness_values.append(value)
    else:
        fitness_values.append(0)

sorted_fitness = sorted(fitness_values)
print(sorted_fitness)
print(coordinates)

for i in range(count):
    if sorted_fitness[count-1] == fitness_values[i]:
        max_point = coordinates[i]

fig,ax = plt.subplots(1)
ax.imshow(i2, cmap= 'gray')
rect = patches.Rectangle(max_point,29,35,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.xticks(rotation=90)
plt.show()

# GRAPH TO SEE THE RELATION
# fitness_values = np.asarray(fitness_values)
print(generations)
plt.plot(generations, fitness_values, label = 'max')
plt.title('Fitness against each generation')
plt.xlabel('No. of Generations')
plt.ylabel('Max Fitness')
plt.show()

