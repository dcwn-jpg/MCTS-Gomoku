### MCTS AGENT (PUCT)
from __future__ import print_function
from gomoku_env import GomokuEnv
import time
import sys
from collections import deque, defaultdict
import numpy as np
from numpy import random, sqrt, argwhere, zeros


###=== PARAMETERS CHANGES MADE ===###
###     N_SIMUL = 800 --> 500     ###
###     GAME = 5 --> 1            ###
###     c_pucb = 5 --> 1          ###
###     BOARD_SIZE = 9 --> 7      ###
###===============================###


### MCTS & GAME PARAMETERS (this overrides the parameters declared in gomoku_env)
## N = Visits, Q = Average Value[ (win_result - Q) / N ] {both values update over the duration of a game}
N, Q = 0, 1
## Game Parameters
CURRENT = 0
OPPONENT = 1
COLOR = 2
BLACK = 1
WHITE = 0
BOARD_SIZE = 9      # (Original value of 9)
HISTORY = 2
## Number of iterations per move  (Original value of 800)
N_SIMUL = 800
## Total number of games to be played  (Original value of 5)
GAME = 5


class MCTS:
    def __init__(self, board_size, n_history, n_simul):
        # display=false as these are used for the rollouts in Phase 3.
        self.env_simul = GomokuEnv(board_size, n_history, display=False)
        self.board_size = board_size
        self.n_simul = n_simul
        # Filled by reset_tree()
        self.tree = None
        self.root = None                # Copy of currently searched real state
        self.state = None               # Current iterated state
        self.board = None               # Current iterated board
        self.key_memory = deque()       # States visited by this simulation (used for backup)
        self.action_memory = deque()    # Moves taken at said state
        self.reset_tree()

    def reset_tree(self):
        # Dictionary used for state-hash of each node (81x2 array [visits{N}, average_value{Q}]) 
        self.tree = defaultdict(lambda: zeros((self.board_size**2, 2)))     # Missing key creates a zero array

    ## Function to "think" about what move to play
    def get_action(self, state, board):
        self.root = state.copy()
        self._simulation(state)                         # Builds tree up to N_SIMUL iterations
        self.board = board                              # Init root board after playout iterations overwrote it
        # Root state's dictionary key
        root_key = hash(self.root.tobytes())            # Updated deprecated NumPy func from .tostring() --> .tobytes()
        # Argmax Q or Argmin Q
        action = self._selection(root_key, c_pucb=0)    # Determines which move to pick, c=0, (most-visited move)
        return action

    ## EXPANSION & ROLLOUTS(SIMULATION) FUNCTION [_simulation()]
    def _simulation(self, state):
        start = time.time()
        finish = 0
        ## 0. Start Rollouts/Simulations
        for sim in range(self.n_simul):
            print('\rsimulation: {}'.format(sim + 1), end='')       # Progress counter
            sys.stdout.flush()
            # Reset state back to root position after each rollout
            self.state, self.board = self.env_simul.reset(state)
            done = False
            is_expansion = True                                     # Turns False after first new state
            ## 3. ROLLOUTS(SIMULATION) PHASE [Plays one full simulated game]
            while not done:
                # Identify current state using dictionary hash
                key = hash(self.state.tobytes())                    # Updated deprecated NumPy func from .tostring() --> .tobytes() 
                ## 1. SELECTION PHASE (Searches the tree)
                if key in self.tree:
                    # Calls PUCT based _selection() with set exploration value[c_pucb={1}] (Original value of 5)
                    action = self._selection(key, c_pucb=5)
                    # Remember for backup
                    self.action_memory.appendleft(action)
                    self.key_memory.appendleft(key)
                else:
                    ## 2. EXPANSION PHASE then continue back to ROLLOUT
                    legal_move, _ = self._get_legal_move(self.board)
                    action = random.choice(legal_move)              # Expands based on a uniformly random legal move
                    if is_expansion:
                        self.action_memory.appendleft(action)
                        self.key_memory.appendleft(key)
                        is_expansion = False                        # Only first new state is recorded
                ## Finally makes the move [calls step()]
                self.state, self.board, reward, done = self.env_simul.step(action)
            # done is always True here to ensure backpropagation & backup
            if done:
                ## 4. BACKPROPAGATION PHASE (Backup & reset memory)
                self._backup(reward)
                finish = time.time() - start
                # Unused code (Assume it was made for testing purposes)
                # if finish >= self.think_time:
                #     break
        print('\r{} simulations end ({:0.0f}s)'.format(sim + 1, finish))

    def _get_legal_move(self, board):
        board_fill = board[CURRENT] + board[OPPONENT]       # 1 = cell filled
        legal_move = argwhere(board_fill != 1).flatten()    # Indexes of empty cells
        return legal_move, board_fill

    ## SELECTION FUNCTIONS [_selection() & _get_pucb()]
    def _selection(self, key, c_pucb):
        edges = self.tree[key]                              # Gets current state's N & Q in table form
        pucb = self._get_pucb(edges, c_pucb)                # PUCT score for each cell
        # Final move decision at root (when using UCT, in PUCT's case the next selection statement handles final move)
        if c_pucb == 0:
            visit = edges[:, N]
            print('\nvisit count')
            print(visit.reshape(self.board_size, self.board_size).round())
            # Gets all moves tied for most visits then randomly breaks them
            action = argwhere(visit == visit.max()).flatten()
            action = action[random.choice(len(action))]
            return action
        # Determine's who turn it is
        if self.board[COLOR][0] == WHITE:
            # Black's choice (takes highest score)
            action = argwhere(pucb == pucb.max()).flatten()
        else:
            # White's choice (takes lowest score)
            action = argwhere(pucb == pucb.min()).flatten()
        action = action[random.choice(len(action))]
        return action

    ## PUCT FORMULA CONFIGURATION       |       PUCT = Q + c * prior * √(total visits) / (1 + visits)
    def _get_pucb(self, edges, c_pucb):
        legal_move, no_legal_loc = self._get_legal_move(self.board)
        # THE PRIOR PARAMETER ( Originally uniform [ 1/len(legal_move) ] )
        prior = 1/len(legal_move)
        total_N = edges.sum(0)[N]           # Total N across ALL moves
        # Black's pucb
        if self.board[COLOR][0] == WHITE:
            no_legal_loc *= -9999           # Occupied Cells = -9999 are never chosen by Argmax Q
            ##== PUCT FORMULA (for black {+}) ==## 
            pucb = edges[:, Q] + c_pucb * prior * sqrt(total_N) / (edges[:, N] + 1) + no_legal_loc
        # White's pucb
        else:
            no_legal_loc *= 9999            # Occupied Cells = 9999 are never chosen by Argmin Q
            ##== PUCT FORMULA (for white {-}) ==##
            pucb = edges[:, Q] - c_pucb * prior * sqrt(total_N) / (edges[:, N] + 1) + no_legal_loc
        return pucb

    ## BACKPROPAGATION FUNCTION
    def _backup(self, reward):
        # Walks recorded path then update edges in the tree
        while self.action_memory:
            key = self.key_memory.popleft()
            action = self.action_memory.popleft()
            edges = self.tree[key]                  # Creates edge entry (if new)
            edges[action][N] += 1                   # Increments Visit (N)
            edges[action][Q] += (reward - edges[action][Q]) / edges[action][N]  # Average Value (Q)
        return 0

### Runs a real Gomoku environment with 1 MCTS Agent repeatedly based on GAME parameter
def play():
    env = GomokuEnv(BOARD_SIZE, HISTORY)
    mcts = MCTS(BOARD_SIZE, HISTORY, N_SIMUL)
    result = {'Black': 0, 'White': 0, 'Draw': 0}
    ## Loops render(), get_action(), step() until game end
    for g in range(GAME):
        print('#' * (BOARD_SIZE - 4),
              ' GAME: {} '.format(g + 1),
              '#' * (BOARD_SIZE - 4))
        # If game done, reset state 
        state, board = env.reset()
        done = False
        # If game not done continue loop
        while not done:
            env.render()
            # Start simulations
            action = mcts.get_action(state, board)
            state, board, z, done = env.step(action)
        if done:
            if z == 1:
                result['Black'] += 1
            elif z == -1:
                result['White'] += 1
            else:
                result['Draw'] += 1
            # Render & reset tree
            env.render()
            mcts.reset_tree()
        # Result is tallied from z
        # Winrate is not the true win-loss rate and can be simplified to the sigmoid function:
        # 100 / (1 + e^((whw - blw) / k)), where k = (g + 1)
        # Note: If black wins every game, the winrate will be 73.1% & 26.9% if white does
        print('')
        print('=' * 20, " {}  Game End  ".format(g + 1), '=' * 20)
        blw, whw, drw = result['Black'], result['White'], result['Draw']
        stat = (
            'Black Win: {}  White Win: {}  Draw: {}  Winrate: {:0.1f}%'.format(
                blw, whw, drw,
                1 / (1 + np.exp(whw / (g + 1)) / np.exp(blw / (g + 1))) * 100))
        print(stat, '\n')


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    #np.random.seed(0)
    play()
