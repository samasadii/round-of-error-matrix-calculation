from copy import copy, deepcopy
import numpy as np

def get_matrix(dimention):
    matrices = []
    for i in range(3):
        line = input().split(' ')
        if (i < 2):
            matrix = [] 
            index = 0
            for j in range(dimention):
                temp = []
                for k in range(dimention):
                    temp.append(float(line[index]))
                    index += 1
                matrix.append(temp[:])
        elif (i == 2):
            matrix = [] 
            for j in range(dimention):
                matrix.append([float(line[j])])
        matrices.append(matrix)
    return matrices

def get_input():
    samples = []
    dimention = int(input())
    while(dimention > 0):
        sample = get_matrix(dimention)
        samples.append(sample)
        dimention = int(input())
    return samples

def determinant(matrix):
    
    total = 0
    indices = list(range(len(matrix)))

    if len(matrix) == 2 and len(matrix[0]) == 2:
        val = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        return val
 
    for fc in indices:
        temp = deepcopy(matrix)
        temp = temp[1:]
        height = len(temp)
        for i in range(height): 
            temp[i] = temp[i][0:fc] + temp[i][fc+1:] 
 
        sign = (-1) ** (fc % 2)
        sub_det = determinant(temp)
        total += sign * matrix[0][fc] * sub_det 
 
    return total

def inverse(A):

    n = len(A)
    AM = deepcopy(A)
    I = np.identity(n)
    I = I.tolist()

    indices = list(range(n)) 
    for fd in range(n):
        fd_scaler = 1.0 / AM[fd][fd] 
        for j in range(n):
            AM[fd][j] *= fd_scaler
            I[fd][j] *= fd_scaler

        for i in indices[0:fd] + indices[fd+1:]: 
            cr_scaler = AM[i][fd]
            for j in range(n): 
                AM[i][j] = AM[i][j] - cr_scaler * AM[fd][j]
                I[i][j] = I[i][j] - cr_scaler * I[fd][j]

    return I

def multiply(a, b):
    result = np.zeros(shape=(len(a),len(b[0]))).tolist()
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result

def solve_x(a, b):
    inverse_a = inverse(a)
    return multiply(inverse_a, b)

def residual_vector(a, b, x):
    temp = multiply(a, x)
    return np.subtract(b, temp).tolist()

def norm(a):
    ssum = 0
    for m in a:
        for i in m:
            ssum += i ** 2

    return ssum ** 1/2
    
def difference(a, b):
    return a - b


def run():
    samples = get_input()
    for sample in samples:
        if (determinant(sample[0]) != 0):
            det_a = determinant(sample[0])
            inv_a = inverse(sample[0])
            inv_h = inverse(sample[1])
            x_a = solve_x(sample[0], sample[2])
            x_h = solve_x(sample[1], sample[2])
            residual_a = residual_vector(sample[0], sample[2], x_a)
            residual_h = residual_vector(sample[1], sample[2], x_h)
            norm_a = norm(residual_a)
            norm_h = norm(residual_h)
            comp = difference(norm_a, norm_h)
            print("Matrix A: \n{}\nMatrix H: \n{}\nMatrix b: \n{}\n".format(np.asmatrix(sample[0]), np.asmatrix(sample[1]), np.asmatrix(sample[2])))
            print("Det A: {}".format(det_a))
            print("inverse A: \n{}\ninverse H: \n{}".format(np.asmatrix(inv_a), np.asmatrix(inv_h)))
            print("Ax = b \nx: \n{}\nHx = b \nx: \n{}".format(np.asmatrix(x_a), np.asmatrix(x_h)))
            print("b-Ax: \n{}\nb-Hx: \n{}".format(np.asmatrix(residual_a), np.asmatrix(residual_h)))
            print("||b-Ax||: {}\n||b-Hx||: {}".format(norm_a, norm_h))
            print("||b-Ax|| - ||b-Hx||: {}".format(comp))
            


run()