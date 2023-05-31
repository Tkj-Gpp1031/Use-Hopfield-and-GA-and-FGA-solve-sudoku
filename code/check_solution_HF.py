def is_valid_sudoku(board):
    def is_valid_row(row):
        unique_row = [num for num in row if num != 0]
        return len(set(unique_row)) == len(unique_row)

    def is_valid_column(col):
        unique_col = [board[row][col] for row in range(9) if board[row][col] != 0]
        return len(set(unique_col)) == len(unique_col)

    def is_valid_box(start_row, start_col):
        unique_box = []
        for row in range(3):
            for col in range(3):
                num = board[start_row + row][start_col + col]
                if num != 0:
                    unique_box.append(num)
        return len(set(unique_box)) == len(unique_box)

    for i in range(9):
        return is_valid_row(board[i])
    for i in range(9):
        return is_valid_column(i)


    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            return is_valid_box(i, j)


    return False



with open("data/HF_result_dia.txt", "r") as file:
    f = file.readlines()
    true_count = 0
    for i in range(0,len(f),9):
        sudoku = f[i:i+9]
        puzzle = []
        for j in range(9):
            temp = []
            temp_lst = []
            if j == 0:
                temp = sudoku[j][3:20].split(' ')

                for k in range(9):
                    temp_lst.append(int(temp[k]))
                puzzle.append(temp_lst)
            else:
                temp = sudoku[j][2:19].split(' ')
                for k in range(9):
                    temp_lst.append(int(temp[k]))
                puzzle.append(temp_lst)

        if is_valid_sudoku(puzzle):
            true_count += 1
    print(true_count/100)