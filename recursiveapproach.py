import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(1500)

img1 = mpimg.imread('faceGray.jpg')
img2 = mpimg.imread('groupGray.jpg')

 
# INITIALIZATION OF POPULATION TABLE
def popTableInitialization(pop_size):
    pop_table = []
    for i in range(pop_size):
        pop_table.append((np.random.randint(img2.shape[0]), np.random.randint(img2.shape[1])))
    frameCreation(pop_table, img2) 


# CREATING FRAMES FOR PIXEL MATCHING
def frameCreation(pop_table, img2):
    frames = []
    img2 = np.array(img2)
    for i in range(pop_size):
        frame = img2[pop_table[i][0]:(pop_table[i][0]+img1.shape[0]),pop_table[i][1]:(pop_table[i][1]+img1.shape[1])]
        frames.append(frame)
    fitnessFunction(frames, pop_table)
    

# CORRELATION FUNCTION
def fitnessFunction(frames, pop_table):
    fitness_values = []
    for i in range(pop_size):
        # NUMERATOR
        a = frames[i]-np.mean(frames[i])
        b = img1-np.mean(img1)
        if (a.shape==b.shape):
            mul = a*b    
            mul_sum = np.sum(mul)
        # dENOMINATOR
            a_sq = a**2
            b_sq = b**2
            a_sum = np.sum(a_sq)
            b_sum = np.sum(b_sq)
            result= np.sqrt(a_sum*b_sum)
        # CROSS-CORRELATION
            value = mul_sum/result
            fitness_values.append(value)
        else:
            fitness_values.append(0)
    sortedFitnessValues(fitness_values, pop_table)



# SORTING
def sortedFitnessValues(fitness_values, pop_table):
    sorted_fitness = sorted(fitness_values)
    faceDetection(sorted_fitness, fitness_values, img2, pop_table, threshold= 0.9)


# EXTRACTING THE MAX COORDINATE
def faceDetection(sorted_fitness, fitness_values, img2, pop_table, threshold):
    new_pop_table = sortedPopTable(sorted_fitness, fitness_values, pop_size, pop_table)
    if (sorted_fitness[pop_size - 1] >= threshold):
    # SQUARING THE FACE
        fig,ax = plt.subplots(figsize=(1024, 1024))
        ax.imshow(img2, cmap= 'gray')
        rect = patches.Rectangle(pop_table[0],35,29,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        plt.show()
    else:
        crossOverOfBits(pop_size, new_pop_table)


# MAKING NEW POPULATION TABLE
def sortedPopTable(sorted_fitness, fitness_values, pop_size, pop_table):
    new_pop_table = []
    for i in range(pop_size):
        for j in range(pop_size):
            if (sorted_fitness[i]==fitness_values[j]):
                fitness_values[j] = -100
                new_pop_table.append(pop_table[j])
    return new_pop_table

# CROSS-OVER
def crossOverOfBits(pop_size, new_pop_table):
    binary_number = []
    out_array = np.asarray(new_pop_table, dtype=np.int)
    for i in out_array:
        x = np.binary_repr(i[0], width= 11)
        y = np.binary_repr(i[1], width= 11)
        binary_number.append((x,y))
    concat_binary = [''.join(i) for i in binary_number]
    result_list = []
    for i in range(0,pop_size, 2):
        binary_value_1 = concat_binary[i]
        binary_value_2 = concat_binary[i+1]
        binary_value_1 = list(binary_value_1)
        binary_value_2 = list(binary_value_2)
        random_point = np.random.randint(0, 22)
        for j in range(random_point, 22):
            binary_value_1[j], binary_value_2[j] = binary_value_2[j], binary_value_1[j]
        binary_value_1= ''. join(binary_value_1)
        binary_value_2= ''. join(binary_value_2)
        result_list.append(binary_value_1)
        result_list.append(binary_value_2)
    mutationOfBits(result_list)


# MUTATION
def mutationOfBits(result_list):
    mutated_result=[]
    for value in result_list:
        value= list(value)
        mutation_point = np.random.randint(0,22)
        if(value[mutation_point]==0):
            value[mutation_point]=1
        else:
            value[mutation_point]=0

        value= ''.join(str(v) for v in value)
        mutated_result.append(value)

    newPopTable(mutated_result, img2)


# NEW POP TABLE 
def newPopTable(mutated_result, img2):
    coordinates= []
    for i in range(pop_size):
        coordinates.append((int(mutated_result[i][:11], 2),int(mutated_result[i][12:], 2)))

    frameCreation(coordinates, img2)


pop_size = 50  
pop_table = popTableInitialization(pop_size)



