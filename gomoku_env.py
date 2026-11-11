### GOMOKU SIMULATOR
from __future__ import print_function
from collections import deque
import numpy as np

### GAME PARAMETERS
## Row Numbers (3 Layers) to represent the board in the flat vector  
CURRENT = 0
OPPONENT = 1
COLOR = 2
## Color codes stored in color layer
BLACK = 1
WHITE = 0
## Rendering purposes
COLOR_DICT = {1: 'Black', 0: 'White'}
ALPHABET = 'A B C D E F G H I J K L M N O'


class GomokuEnv:
    def __init__(self, board_size, n_history, display=True):
        self.board_size = board_size
        self.n_history = n_history
        self.display = display
        self.state = None                           # flat 405 state vector
        self.board = None                           # 3 x 81 working board
        self.history = None
        self.done = None
        self.action = None

    ## MCTS Agent calls reset() & step() functions only
    def reset(self, state=None):
        # Initializes state (new game)
        if state is None:
            # 405 state vector creation (n_history{2} * 2 + 1) * board_size{9}^2 = state{405}
            self.state = np.zeros(((self.n_history * 2 + 1) * self.board_size**2), 'float')
            # Creates snapshots of necessary variables at neweest index 0 (Note, after push, snapshot 0 is mover's stone & snapshot 1.. is other player's stones)
            self.history = deque([np.zeros((self.board_size**2), 'float')] * self.n_history * 2, maxlen=self.n_history * 2)
            # 3 "rows" (CURRENT, OPPONENT, COLOR)
            self.board = np.zeros((3, self.board_size**2), 'float')
            self.action = None
        # Passes current state to the simulation's root
        else:
            self.state = state.copy()
            # Reshapes array to 5 layers of 81 cells
            state_origin = self.state.reshape(self.n_history * 2 + 1, self.board_size**2)
            self.history = deque([state_origin[i]
                                  for i in range(self.n_history * 2)],
                                 maxlen=self.n_history * 2)
            self.board = np.zeros((3, self.board_size**2), 'float')
            self.board[CURRENT] = state_origin[1]                   # Layer 1 (Stones of player to move)
            self.board[OPPONENT] = state_origin[0]                  # Layer 0 (Stones of the last mover)
            self.board[COLOR] = state_origin[self.n_history * 2]    # Layer 4 (Color of last mover)
            self.action = None
        return self.state, self.board

    def step(self, action):
        self.action = action
        # Board (Unpacks state again)
        state_origin = self.state.reshape(self.n_history * 2 + 1, self.board_size**2)
        self.board = np.zeros((3, self.board_size**2), 'float')
        self.board[CURRENT] = state_origin[1]
        self.board[OPPONENT] = state_origin[0]
        self.board[COLOR] = state_origin[self.n_history * 2]
        # Checks if a cell is occupied (1 where occupied)
        self.board_fill = (self.board[CURRENT] + self.board[OPPONENT])
        if self.board_fill[self.action] == 1:
            raise ValueError("No Legal Move!")
        # Action (Places the stone)
        self.board[CURRENT][self.action] = 1
        # Snapshots
        self.history.appendleft(self.board[CURRENT])
        # Flips 0 <--> 1 (Mover's color)
        self.board[COLOR] = abs(self.board[COLOR] - 1)
        # New state initialization
        self.state = np.r_[np.asarray(self.history).flatten(), np.asarray(self.board[COLOR]).flatten()]
        # Returns state, board, reward, done values
        return self._check_win(self.board[CURRENT].reshape(self.board_size, self.board_size), self.display)

    ## Helper functions
    def _check_win(self, board, display=True):
        # Throwaway initialization
        current_grid = np.zeros((5, 5))
        # Check for each possible win scenario in a 5x5 window position
        for row in range(self.board_size - 5 + 1):
            for col in range(self.board_size - 5 + 1):
                current_grid = board[row: row + 5, col: col + 5]                # The window (based on mover's stones)
                sum_horizontal = np.sum(current_grid, axis=1)                   # 5 Row totals (-)
                sum_vertical = np.sum(current_grid, axis=0)                     # 5 Column totals (|)
                sum_diagonal_1 = np.sum(current_grid.diagonal())                # Main diagonal (\)
                sum_diagonal_2 = np.sum(np.flipud(current_grid).diagonal())     # Opposing diagonal (/)
                # Checks if stones are in full row/column of 5
                if 5 in sum_horizontal or 5 in sum_vertical:
                    done = True
                    color = self.board[COLOR][0]    # Already flipped
                    if color == BLACK:
                        reward = 1
                    else:
                        reward = -1
                    if display:
                        print('\n#########   {} Win!   #########'.format(COLOR_DICT[color]))
                    return self.state, self.board, reward, done
                # Diagonals has separate selection statement here with same logic for cleaner code (?)
                if sum_diagonal_1 == 5 or sum_diagonal_2 == 5:
                    reward = 1
                    done = True
                    color = self.board[COLOR][0]
                    if color == BLACK:
                        reward = 1
                    else:
                        reward = -1
                    if display:
                        print('\n#########   {} Win!   #########'.format(COLOR_DICT[color]))
                    return self.state, self.board, reward, done
        # Checks for a draw scenario (80 stones put down with no full 5 consecutive same colored stones)
        if np.sum(self.board_fill) == self.board_size**2 - 1:
            reward = 0
            done = True
            if display:
                print('\n#########     Draw!     #########')
            return self.state, self.board, reward, done
        else:  # Continue
            reward = 0
            done = False
            return self.state, self.board, reward, done

    # Just renders the current state of the board
    def render(self):
        action_coord = None
        action_right = None
        if self.action is not None:
            if (self.action + 1) % self.board_size == 0:
                action_right = None
            else:
                action_right_x = (self.action + 1) // self.board_size
                action_right_y = (self.action + 1) % self.board_size
                action_right = (action_right_x, action_right_y)
            action_coord_x = self.action // self.board_size
            action_coord_y = self.action % self.board_size
            action_coord = (action_coord_x, action_coord_y)
        if self.board[COLOR][0] == BLACK:
            board = (self.board[CURRENT] + self.board[OPPONENT] * 2).reshape(
                self.board_size, self.board_size)
        else:
            board = (self.board[CURRENT] * 2 + self.board[OPPONENT]).reshape(
                self.board_size, self.board_size)
        count = np.sum(self.board[CURRENT] + self.board[OPPONENT])
        board_str = '\n   ' + ALPHABET[:self.board_size * 2 - 1] + '\n'
        for i in range(self.board_size):
            for j in range(self.board_size):
                if j == 0:
                    board_str += '{:2}'.format(i + 1)
                if board[i][j] == 0:
                    if (i, j) == action_right:
                        board_str += '.'
                    else:
                        board_str += ' .'
                if board[i][j] == 1:
                    if (i, j) == action_coord:
                        board_str += '(O)'
                    elif (i, j) == action_right:
                        board_str += 'O'
                    else:
                        board_str += ' O'
                if board[i][j] == 2:
                    if (i, j) == action_coord:
                        board_str += '(X)'
                    elif (i, j) == action_right:
                        board_str += 'X'
                    else:
                        board_str += ' X'
                if j == self.board_size - 1:
                    board_str += '\n'
        board_str += '  ' + '-' * (self.board_size - 6) + \
            '  MOVE: {:2.0f}  '.format(count) + '-' * (self.board_size - 6)
        print(board_str)

## Testing block if file is ran by itself (it will play 3 test moves and print the board after each move)
if __name__ == '__main__':
    env = GomokuEnv(9, 2)
    env.reset()
    env.step(40)
    env.render()
    env.step(41)
    env.render()
    env.step(49)
    env.render()
