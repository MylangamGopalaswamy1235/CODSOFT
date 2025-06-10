import math

# Initialize the board
board = [' ' for _ in range(9)]

# Print the board
def print_board():
    for row in [board[i*3:(i+1)*3] for i in range(3)]:
        print('| ' + ' | '.join(row) + ' |')

# Check if a player has won
def check_winner(player):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

# Check for tie
def is_full():
    return ' ' not in board

# Minimax algorithm
def minimax(is_maximizing):
    if check_winner('O'):
        return 1
    elif check_winner('X'):
        return -1
    elif is_full():
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                score = minimax(False)
                board[i] = ' '
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                score = minimax(True)
                board[i] = ' '
                best_score = min(score, best_score)
        return best_score

# AI move
def ai_move():
    best_score = -math.inf
    move = None
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            score = minimax(False)
            board[i] = ' '
            if score > best_score:
                best_score = score
                move = i
    board[move] = 'O'

# Human move
def human_move():
    while True:
        try:
            pos = int(input("Enter your move (1-9): ")) - 1
            if 0 <= pos <= 8 and board[pos] == ' ':
                board[pos] = 'X'
                break
            else:
                print("Invalid move. Try again.")
        except:
            print("Please enter a number from 1 to 9.")

# Main game loop
def play_game():
    print("Welcome to Tic-Tac-Toe! You are X, AI is O.")
    print_board()

    while True:
        human_move()
        print_board()
        if check_winner('X'):
            print("You win!")
            break
        elif is_full():
            print("It's a tie!")
            break

        print("AI's turn...")
        ai_move()
        print_board()
        if check_winner('O'):
            print("AI wins!")
            break
        elif is_full():
            print("It's a tie!")
            break

# Start the game
play_game()