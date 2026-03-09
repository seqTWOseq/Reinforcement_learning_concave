import numpy as np

BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

def create_board():
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

def print_board(board):
    print("   " + " ".join(f"{i:2}" for i in range(BOARD_SIZE)))
    for i, row in enumerate(board):
        row_str = f"{i:2} "
        for cell in row:
            if cell == EMPTY:
                row_str += " ."
            elif cell == BLACK:
                row_str += " X"
            else:
                row_str += " O"
        print(row_str)
    print()

def check_winner(board, row, col, player):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        for sign in [1, -1]:
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == player:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= 5:
            return True
    return False

def get_move(player_name):
    while True:
        try:
            inp = input(f"{player_name}의 차례 (행 열 입력, 예: 7 7): ").strip()
            row, col = map(int, inp.split())
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                return row, col
            print(f"0~{BOARD_SIZE - 1} 범위 안에서 입력하세요.")
        except (ValueError, TypeError):
            print("올바른 형식으로 입력하세요 (예: 7 7)")

def play():
    board = create_board()
    players = {BLACK: "흑(X)", WHITE: "백(O)"}
    current = BLACK

    print("=== 오목 게임 ===")
    print(f"15x15 보드, 5개를 먼저 연속으로 놓으면 승리!")
    print()

    for turn in range(BOARD_SIZE * BOARD_SIZE):
        print_board(board)
        row, col = get_move(players[current])

        if board[row][col] != EMPTY:
            print("이미 돌이 놓인 자리입니다. 다시 입력하세요.")
            continue

        board[row][col] = current

        if check_winner(board, row, col, current):
            print_board(board)
            print(f"{players[current]} 승리!")
            return

        current = WHITE if current == BLACK else BLACK

    print_board(board)
    print("무승부!")

if __name__ == "__main__":
    play()
