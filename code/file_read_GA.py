import os

import numpy as np
with open('data/easy.txt', encoding='utf-8') as file_obj:
    contents = file_obj.readlines()
    sudoku_puzzle = []
    all_sudoku = []
    temp = []
    col = 1
    row = 0
    for puzz in contents:

        for i in puzz[13:94]:

            temp.append(int(i))

            if col == 9:
                sudoku_puzzle.append(temp)
                row += 1
                col = 0
                temp = []
            col += 1
            if row == 9:
                all_sudoku.append(sudoku_puzzle)
                sudoku_puzzle = []
                row = 0





