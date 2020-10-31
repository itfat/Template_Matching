import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
# import random
# from numpy import cov

img2 = mpimg.imread('groupGray.jpg')
# print(img2)
# print(img2.shape)
# imgplot = plt.imshow(img2, cmap= 'gray')
# plt.show()

img1 = mpimg.imread('boothiGray.jpg')
# print(img1)
# print(img1.shape)
# imgplot = plt.imshow(img1, cmap='gray')
# plt.show()

pop_size = 50   
# random.seed(1)
pop_table = []
# p1 = np.random.randint(img2.shape[0], size=pop_size)
# p2 = np.random.randint(img2.shape[1], size=pop_size)
# pop_table = [(p1[i], p2[i]) for i in range(0,pop_size)]
for i in range(pop_size):
    pop_table.append((np.random.randint(img2.shape[0]), np.random.randint(img2.shape[1])))
# print(pop_table)
# print("size of pop is: ", len(pop_table))
# print(pop_table[0][0])
# print(pop_table[0][0]+img1.shape[0])
# print(pop_table[0][1])
# print(pop_table[0][1]+img1.shape[1])
# pop_table = np.pop_table
img2 = np.array(img2)
fitness_values = []
for i in range(pop_size):
    frame = img2[pop_table[i][0]:(pop_table[i][0]+img1.shape[0]),pop_table[i][1]:(pop_table[i][1]+img1.shape[1])]
    # PASS TO CORRELATION FUNCTION
    # NUMERATOR
    a = frame-np.mean(frame)
    b = img1-np.mean(img1)
    if (a.shape==b.shape):
        mul = a*b    
        mul_sum = np.sum(mul)
    # print(mul_sum)
    # dENOMINATOR
        a_sq = a**2
        b_sq = b**2
        a_sum = np.sum(a_sq)
        b_sum = np.sum(b_sq)
        result= np.sqrt(a_sum*b_sum)
    # print(result)
    # CROSS-CORRELATION
        value = mul_sum/result
    # print(value)
        fitness_values.append(value)
    else:
        fitness_values.append(0)
    # print(frame)
# print("frame shape is")

# print(fitness_values)
threshold = 0.9
# SORTING
# print('Sorted')
sorted_fitness = sorted(fitness_values)
# print(sorted_fitness)
# EXTRACTING THE MAX COORDINATE
if (sorted_fitness[pop_size - 1] >= 0.4):
# SQUARING THE FACE
    fig,ax = plt.subplots(1)
    ax.imshow(img2, cmap= 'gray')
    rect = patches.Rectangle(pop_table[i],35,29,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()
# MAKING NEW POPULATION TABLE
new_pop_table = []
for i in range(pop_size):
    for j in range(pop_size):
        if (sorted_fitness[i]==fitness_values[j]):
            fitness_values[j] = -100
            new_pop_table.append(pop_table[j])
    #     print(fitness_values[i])
    #     print(pop_table[i])
   
# print(new_pop_table)
binary_number = []
# print(len(new_pop_table))
out_array = np.asarray(new_pop_table, dtype=np.int)
for i in out_array:
    x = np.binary_repr(i[0], width= 11)
    y = np.binary_repr(i[1], width= 11)
    binary_number.append((x,y))
# print("binary =================================")
# print(binary_number)
# print("uihduhd=----")
# print(binary_number[0][0], binary_number[0][1])
# print(len(binary_number))
concat_binary = [''.join(i) for i in binary_number]
# print(len(concat_binary))
# print(concat_binary)
# result.append(([''.join(x) for x in binary_number[0]]))
# print(result)
result_list = []
for i in range(0,pop_size, 2):
    binary_value_1 = concat_binary[i]
    binary_value_2 = concat_binary[i+1]
    binary_value_1 = list(binary_value_1)
    binary_value_2 = list(binary_value_2)
    # print(binary_value_1, binary_value_2)
    # print(type(binary_value_1[0]),type(binary_value_2))
    random_point = np.random.randint(0, 22)
    # print("CrossOver Point", random_point)
    for j in range(random_point, 22):
        # print(type(binary_value_1[j]))
        binary_value_1[j], binary_value_2[j] = binary_value_2[j], binary_value_1[j]
    binary_value_1= ''. join(binary_value_1)
    binary_value_2= ''. join(binary_value_2)
    result_list.append(binary_value_1)
    result_list.append(binary_value_2)
# print(result_list)

# MUTATION
mutated_result=[]
for value in result_list:
    value= list(value)
    mutation_point = np.random.randint(0,22)
    # print(value[mutation_point])
    if(value[mutation_point]==0):
        value[mutation_point]=1
    else:
        value[mutation_point]=0
    # print(value)
    value= ''.join(str(v) for v in value)
    mutated_result.append(value)
print(mutated_result)


# def binaryToDecimal(bin): 
# return int(n,2) 


# dec_to_binary(new_pop_table)
# print(binary_number)

# binary_value_1 = concat_binary[0]
# binary_value_2 = concat_binary[1]
# binary_value_1 = list(binary_value_1)
# binary_value_2 = list(binary_value_2)
# print(binary_value_1, binary_value_2)
# print(type(binary_value_1),type(binary_value_2))
# random_point = np.random.randint(0, 22)
# print("CrossOver Point", random_point)
# for i in range(random_point, 22):
#     print(type(binary_value_1[i]))
#     binary_value_1[i], binary_value_2[i] = binary_value_2[i], binary_value_1[i]
# binary_value_1= ''. join(binary_value_1)
# binary_value_2= ''. join(binary_value_2)
#     # result_list.append(binary_value_1, binary_value_2)
# print(binary_value_1, binary_value_2)
    
    # random_number = np.random.randint(0,8,dtype=int)

# def Cross_over(Bnumber):
#     for i in range(0, 50, 2):
#        cross_over_imp(Bnumber[i],Bnumber[i+1])
# Cross_over(binary_number)
# print(result)
# print(p1)
# print(p2)
# random.seed(1)
# pop_table = [(np.random.randint(img2.shape[0]), np.random.randint(img2.shape[1])) for i in range(0,pop_size)] 
# print(pop_table)
# for i in range(pop_size):
    
    # pop_table.append(tuple(choice(np.argwhere(img2))))
    # print(img2[tuple(choice(np.argwhere(img2)))])
# print(pop_table)
# print(img2[pop_table[1]])
# Fitness Function
# for i in range(0, 35):
    # for j in range (0,29):


# pop_table = np.random.rand()

# img2 = mpimg.imread('boothi2.png')
# print(img2)
# print(img2.shape)
# imgplot = plt.imshow(img2)
# plt.show()


















