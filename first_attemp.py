import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np

i1 = mpimg.imread('boothiGray.jpg')

# i1 = mpimg.imread('testboothi.png')
i2 = mpimg.imread('groupGray.jpg')
# FOR PNG
# i1 = i1[:,:,2]
img1 = i1.copy()
img2 = i2.copy()

img1 = img1.T
img2 = img2.T
# print(img2.shape)
# print(i1.shape)


# INITIALIZATION OF POPULATION TABLE
def popTableInitialization(pop_size):
    pop_table = []
    for i in range(pop_size):
        pop_table.append((np.random.randint(img2.shape[0]), np.random.randint(img2.shape[1])))
    return pop_table 


# CREATING FRAMES FOR PIXEL MATCHING
def frameCreation(pop_table, img2):
    frames = []
    img2 = np.array(img2)
    for i in range(pop_size):
        frame = img2[pop_table[i][0]:(pop_table[i][0]+img1.shape[0]),pop_table[i][1]:(pop_table[i][1]+img1.shape[1])]
        frames.append(frame)
    return frames
    

# CORRELATION FUNCTION
# its not giving value greater than 0.5
# now its max is 0.7
def fitnessFunction(frames, pop_table, img1):
    fitness_values = []
    for i in range(pop_size):
        if frames[i].shape == img1.shape:
            frames[i] = frames[i] - frames[i].mean()
            img1 = img1 - img1.mean()
            frames[i] = frames[i] / frames[i].std()
            img1 = img1 / img1.std()
            value = np.mean(frames[i]*img1)
            fitness_values.append(value)
        else:
            fitness_values.append(0)
    return fitness_values



# SORTING
def sortedFitnessValues(fitness_values, pop_table):
    sorted_fitness = sorted(fitness_values)
    return sorted_fitness


# EXTRACTING THE MAX COORDINATE
def faceDetection(sorted_fitness, fitness_values, i2, pop_table, threshold):
    sorted_pop_table = sortedPopTable(sorted_fitness, fitness_values, pop_size, pop_table)
    if (sorted_fitness[pop_size - 1] >= threshold):
    # SQUARING THE FACE
        
        fig,ax = plt.subplots(1)
        ax.imshow(i2, cmap= 'gray')
        rect = patches.Rectangle(sorted_pop_table[pop_size-1],29,35,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        plt.xticks(rotation=90)
        plt.show()
        return []
    else:
        return sorted_pop_table


# MAKING NEW POPULATION TABLE
def sortedPopTable(sorted_fitness, fitness_values, pop_size, pop_table):
    sorted_pop_table = []
    for i in range(pop_size):
        for j in range(pop_size):
            if (sorted_fitness[i]==fitness_values[j]):
                fitness_values[j] = -100
                sorted_pop_table.append(pop_table[j])
    return sorted_pop_table

# CROSS-OVER
def crossOverOfBits(pop_size, new_pop_table):
    binary_number = []
    out_array = np.asarray(new_pop_table, dtype=np.int)
    for i in out_array:
        x = np.binary_repr(i[0], width= 11)
        # print("---------------------------------------------------------")
        # print(x)
        y = np.binary_repr(i[1], width= 11)
        # print(y)
        # print("---------------------------------------------------------")
        binary_number.append((x,y))
    concat_binary = [''.join(i) for i in binary_number]
    # print(concat_binary)
    result_list = []
    for i in range(0,pop_size, 2):
        # print(pop_size)
        binary_value_1 = concat_binary[i]
        binary_value_2 = concat_binary[i+1]
        binary_value_1 = list(binary_value_1)
        binary_value_2 = list(binary_value_2)
        random_point = np.random.randint(1, 22)
        for j in range(random_point, 22):
            binary_value_1[j], binary_value_2[j] = binary_value_2[j], binary_value_1[j]
        binary_value_1= ''. join(binary_value_1)
        binary_value_2= ''. join(binary_value_2)
        result_list.append(binary_value_1)
        result_list.append(binary_value_2)
    return result_list


# MUTATION
def mutationOfBits(result_list):
    mutated_result=[]
    for value in result_list:
        value= list(value)
        mutation_point = np.random.randint(0,22)
        if(value[mutation_point]=='0'):
            # print(True)
            value[mutation_point]='1'
        else:
            value[mutation_point]='0'
        value= ''.join(str(v) for v in value)
        mutated_result.append(value)
    return mutated_result


# NEW POP TABLE 
def newPopTable(mutated_result, img2):
    coordinates= []
    for i in range(pop_size):
        x = int(mutated_result[i][:11], 2)
        y = int(mutated_result[i][12:], 2)
        if x <= 1024 and y <= 512:
            coordinates.append((x,y))
        else:
            coordinates.append((np.random.randint(img2.shape[0]), np.random.randint(img2.shape[1])))

    return coordinates


pop_size = 50
pop_table = popTableInitialization(pop_size)
# print(pop_table)
frames = frameCreation(pop_table, img2)
# print(frames)
fitness_values = fitnessFunction(frames, pop_table, img1)
# print(fitness_values)
sorted_fitness = sortedFitnessValues(fitness_values, pop_table)
print(sorted_fitness)
sorted_pop_table = faceDetection(sorted_fitness, fitness_values, i2, pop_table, threshold= 0.9)
# print(sorted_fitness)
count= 0
generations = []
generations.append(0)
save_max = []
save_max.append(sorted_fitness[pop_size-1])
save_mean = []
save_mean.append(np.mean(sorted_fitness))
for i in range(1,2500):
    # if (sorted_pop_table!=0):
    if sorted_pop_table == []:
        break
    else:
        generations.append(i)
        save_max.append(sorted_fitness[pop_size-1])
        save_mean.append(np.mean(sorted_fitness))
        print(sorted_pop_table)
        result_list = crossOverOfBits(pop_size, sorted_pop_table)
        # print("---------------------------------------------------------")
        # print(result_list)
        mutated_result = mutationOfBits(result_list)
        # print(mutated_result)
        new_pop_table = newPopTable(mutated_result, img2)
        # print(new_pop_table)
        frames = frameCreation(new_pop_table, img2)
        # print(frames)
        fitness_values = fitnessFunction(frames, new_pop_table, img1)
        sorted_fitness = sortedFitnessValues(fitness_values, new_pop_table)
        print(sorted_fitness)
        count=count+1
    
    sorted_pop_table = faceDetection(sorted_fitness, fitness_values, i2, new_pop_table, threshold= 0.9)

print(count)
new = []
new = sorted(save_max)
# print(new)
new2 = []
new2 = sorted(save_mean) 
# PLOT OF THE FITNESS VALUE

plt.scatter(generations, save_max, label = 'max')
plt.scatter(generations, save_mean, label = 'mean')
plt.title('Fitness against each generation')
plt.xlabel('No. of Generations')
plt.legend()
plt.show() 

