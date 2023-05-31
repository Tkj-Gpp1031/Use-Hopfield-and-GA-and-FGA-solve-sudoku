import numpy
import numpy as np
import numpy.random as rnd
import math
from tqdm import tqdm
import time
import file_read as f
import pandas as pd
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def index_ijk(n, d):
    i = math.floor(n / d ** 2)
    j = math.floor((n - (i * d ** 2)) / d)
    k = n - (i * d ** 2) - (j * d)
    return i, j, k


def ijk_index(i, j, k, d):
    n = i * d ** 2 + j * d + k
    return n


#Calculation of weights based on constraints
def getMatAndVec(board, l_h=4, l_r=3, l_c=3, l_b=3, l_s=1):
    dimension = board.shape[0]
    #Store locations with numbers
    hints = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] > 0:
                n = ijk_index(i, j, board[i][j] - 1, dimension)
                hints.append(n)
    #print(len(hints))
    H = np.zeros((len(hints), dimension ** 3))
    for instance in range(len(H)):
        H[instance][hints[instance]] = 1
    #     print('instance:',instance)
    #     print('hints:',hints[instance])
    # print('len_H:',len(H))


    R = np.zeros((dimension ** 2, dimension ** 3))

    for instance in range(len(R)):
        current_i = math.floor(instance / dimension)
        current_k = instance % dimension
        #print('i:',current_i)
        #print('k:',current_k)
        for n in range(len(R[instance])):
            i, j, k = index_ijk(n, dimension)
            if (i == current_i and k == current_k):
                R[instance][n] = 1

    C = np.zeros((dimension ** 2, dimension ** 3))
    for instance in range(len(C)):
        current_j = math.floor(instance / dimension)
        current_k = instance % dimension
        for n in range(len(C[instance])):
            i, j, k = index_ijk(n, dimension)
            if (j == current_j and k == current_k):
                C[instance][n] = 1

    B = np.zeros((dimension ** 2, dimension ** 3))
    for instance in range(len(B)):
        current_i = math.floor(instance / dimension)
        current_j = instance % dimension
        for n in range(len(B[i])):
            i, j, k = index_ijk(n, dimension)
            if (i == current_i and j == current_j):
                B[instance][n] = 1

    S = np.zeros((dimension ** 2, dimension ** 3))
    for instance in range(len(S)):
        region = math.floor(instance / dimension)
        current_k = instance % dimension
        for n in range(len(S[i])):
            i, j, k = index_ijk(n, dimension)
            same_region = (region % 3 == math.floor(j / 3) and (math.floor(region / 3) == math.floor(i / 3)))
            if (same_region and k == current_k):
                S[instance][n] = 1

    v1_n2 = np.ones(dimension ** 2)
    v1_n3 = np.ones(dimension ** 3)
    v1_h = np.ones(len(hints))

    Q = (l_r * R.T @ R) + (l_c * C.T @ C) + (l_b * B.T @ B) + (l_s * S.T @ S) + (l_h * H.T @ H)
    q = 2 * ((l_r * v1_n2 @ R) + (l_c * v1_n2 @ C) + (l_b * v1_n2 @ B) + (l_s * v1_n2 @ S) + (l_h * v1_h @ H))

    W = -0.5 * Q
    np.fill_diagonal(W, 0)
    t = 0.5 * (Q @ v1_n3 - q)#Transpose of the offset vector matrix

    return W, t
#Checking constraints
def check_cell(board, row, col, num):
    # Check row
    if num in board[row]:
        return False

    # Check column
    if num in [board[i][col] for i in range(9)]:
        return False

    # Check box
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False

    return True

#Create implicit check bits
def check_solution_with_implicit(board):
    for i in range(9):
        for j in range(9):
            num = board[i][j]
            if num == 0 or not check_cell(board, i, j, num):
                return False
    return True
'''
The following is a simulated annealing process where the solution matrix and state matrix are updated 
when the updated energy is smaller and the temperature drops. When the updated energy value is not less than the original energy, 
there is a probability that the solution matrix is updated and the temperature drops.
When a smaller energy value is not found several times in a row, it is forced to reduce the temperature.
'''
def simulatedAnnealing(s, W, t, Th=10, Tl=0.5, numT=21, passes=1500, patience=10, cooling_rate_decay=0.9):
    n = len(s)
    best_energy = float("inf")
    best_s = s.copy()

    T = Th
    no_improvement_count = 0
    iter_number = 0
    while T > Tl:
        for r in range(passes):
            iter_number += 1
            for i in range(n):
                q = 1 / (1 + np.exp(-2 / T * (W[i] @ s - t[i])))
                z = rnd.binomial(n=1, p=q)
                s[i] = 2 * z - 1
                energy = (-0.5 * s.T @ W @ s) + (t.T @ s)

            if energy < best_energy:
                best_energy = energy
                best_s = s.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                no_improvement_count = 0
                T *= cooling_rate_decay

    return best_s,best_energy,iter_number
#This part converts the offset vector into a sudoku solution of a two-dimensional matrix.
def state_to_sudoku(s, dimension):
    output = np.zeros((dimension, dimension), dtype=np.int8)
    for n in range(len(s)):
        i, j, k = index_ijk(n, dimension)
        if s[n] == 1:
            output[i, j] = k + 1
    return output
#Converts sudoku to int type.
def stringToBoard(board):
    n = int(np.sqrt(len(board)))
    b = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            b[i][j] = int(board[i * n + j])
    return b
def run(puzz):
    question = str()

    for i in range(9):
        for j in range(9):
            question += str(puzz[i][j])
    board = stringToBoard(question)

    n3 = board.shape[0] ** 3
    matW, vecT = getMatAndVec(board, l_h=4, l_b=2, l_r=1, l_c=1, l_s=1)

    vecS = np.random.binomial(n=1, p=0.05, size=n3) * 2 - 1
    vecS_result,energy,iter_num = simulatedAnnealing(vecS, matW, vecT, Th=10, Tl=0.5, numT=21, passes=1500,patience=10, cooling_rate_decay=0.9)

    result = state_to_sudoku(vecS_result, 9)
    return result,iter_num
sudoku = f.all_sudoku
#write solution
# f = open("data/HF_result_dia.txt", "a")
#
#
# for i in range(len(sudoku)):
#     puzz = sudoku[i]
#     ticks = time.time()
#     result = run(puzz)
#     ticks_end = time.time()
#     f.write(str(result))
#     print(result)
#     print('runing_time:',ticks_end-ticks)
#     if i == 100:
#         break
#
# f.close()
f = open("data/HF_time.txt", "a")
f.write('dia_time\n')
for i in range(len(sudoku)):
    puzz = sudoku[i]
    ticks = time.time()
    result,iter_num = run(puzz)
    ticks_end = time.time()
    total_time = ticks_end-ticks
    f.write(str(total_time)+'\n')
    print('sudoku',i)
    if i == 100:
        break

f.close()
print('finsh')
