from ..game import Game
from .santorinigo import environment

import numpy as np


class SantoriniGame(Game):
    def __init__(self):
        # we use winning floor = 2 for faster training
        self.env = environment.Santorini((5, 5),
                                         starting_parts=[0, 22, 18, 14, 18],
                                         winning_floor=2)

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        # implement this function
        # look for inspiration from Othello which is provided in full!
        return self.env.reset()

    def get_env(self):
        return self.env

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        # implement this function
        state_shape = self.env.get_state().shape
        return (state_shape[1],state_shape[2])

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.env.action_dim

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        # implement this function
        winning_floor = 2
        parts = board[2]
        starting_parts = [parts[0][0],parts[1][1],parts[2][2],parts[3][3],parts[4][4]]
        size = (board[0].shape[0],board[0].shape[1])
        temp_env =  environment.Santorini(size,
                                         starting_parts=starting_parts,
                                         winning_floor=winning_floor)
        temp_env.set_state(board,player)

        next_state,reward,done,current_player = temp_env.step(action)
        
        return next_state,current_player


    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        winning_floor = 2
        parts = board[2]
        starting_parts = [parts[0][0],parts[1][1],parts[2][2],parts[3][3],parts[4][4]]
        size = (board[0].shape[0],board[0].shape[1])
        temp_env =  environment.Santorini(size,
                                         starting_parts=starting_parts,
                                         winning_floor=winning_floor)
        temp_env.set_state(board,player)
        action_size = self.getActionSize()
        legal = temp_env.legal_moves()
        ans = []
        for i in range(action_size):
            if(i in legal):
                ans.append(1)
            else:
                ans.append(0)
        return np.array(ans)


    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        # implement this function
        winning_floor = 2
        parts = board[2]
        starting_parts = [parts[0][0],parts[1][1],parts[2][2],parts[3][3],parts[4][4]]
        size = (board[0].shape[0],board[0].shape[1])
        temp_env =  environment.Santorini(size,
                                         starting_parts=starting_parts,
                                         winning_floor=winning_floor)
        temp_env.set_state(board,player)
        state = temp_env.get_state()
        building = state[0]
        worker = state[1]

        w1 = np.where(worker == 1)
        w1 = (w1[0][0],w1[1][0])
        w2 = np.where(worker == 2)
        w2 = (w2[0][0],w2[1][0])
        y1 = np.where(worker == -1)
        y1 = (y1[0][0],y1[1][0])
        y2 = np.where(worker == -2)
        y2 = (y2[0][0],y2[1][0])
        if(building[w1] == 2 or building[w2]==2): return 1
        elif(building[y1]==2 or building[y2]==2): return -1

        if(len(temp_env.legal_moves()) == 0 ): return 0.001
        else: return 0    

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
         # implement this function
        winning_floor = 2
        parts = board[2]
        starting_parts = [parts[0][0],parts[1][1],parts[2][2],parts[3][3],parts[4][4]]
        size = (board[0].shape[0],board[0].shape[1])
        temp_env =  environment.Santorini(size,
                                         starting_parts=starting_parts,
                                         winning_floor=winning_floor)
        temp_env.set_state(board,player)
        state = temp_env.get_state()
        state[1] = player*state[1]
        return state
    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        # implement this function
        syms = [(board,pi)]
        # state_shape = self.getBoardSize()
        # w = state_shape[0]
        # h = state_shape[1]
        # line1 = int(w/2)
        # board1 = board.copy()
        # line2 = int(h/2)
        # board2 = board.copy()
        # for j in range(h):
        #     for i in range(line1):
        #         #t = pi1[j][i]
        #         #pi1[j][i] = pi1[j][w-i-1]
        #         #pi1[j][w-i-1] = t
        #         for k in range(2):
        #             b = board1[k][j][i]
        #             board1[k][j][i] = board1[k][j][w-i-1]
        #             board1[k][j][w-i-1] = b
        # syms.append((board1,pi))

        # for j in range(w):
        #     for i in range(line2):
        #         #t = pi2[i][j]
        #         #pi2[i][j] = pi2[h-i-1][j]
        #         #pi2[h-i-1][j] = t
        #         for k in range(2):
        #             b = board2[k][i][j]
        #             board2[k][i][j] = board2[k][h-i-1][j]
        #             board2[k][h-i-1][j] = b
        # syms.append((board2,pi))
        return syms

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        display = []
        for i in range(3):
            for j in range(5):
                for k in range(5):
                    display.append(board[i][j][k])
        return tuple(display)
