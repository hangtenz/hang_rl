import math

import attr
import numpy as np

from .game import Game
from .net import Net

EPS = 1e-8


class MCTS:
    """
    Args: 
        n_sims: number of simulation moves to be made per action selection
        c_puct: the constant in the heuristic
    """
    def __init__(self, game: Game, net: Net, n_sims: int, c_puct: float):
        self.game = game
        self.net = net
        self.n_sims = n_sims
        self.c_puct = c_puct

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, board, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        # implement this function
        # ...
        player = -1
        board = self.game.getCanonicalForm(board,player)
        pi,v = self.net.predict(board)
        valid_move = self.game.getValidMoves(board,player)
        for i in range(len(pi)):
            if(valid_move[i]==0):
               pi[i] = 0
        all_pi = 0
        for i in range(len(pi)):
            all_pi += pi[i]
        pi = pi/all_pi
        return pi

    def search(self, board):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        # implement this function
        # ...
        board = self.game.getInitBoard()
        player = -1
        board = self.game.getCanonicalForm(board,player)
        v = self.game.getGameEnded(board,player)
        while(v==0):
            display = self.game.stringRepresentation(board) 
            valid_move = self.game.getValidMoves(board,player)
            #print(valid_move)
            if(display not in self.Ns):
                self.Ns[display] = 0
            self.Ns[display]+=1
            pi,v = self.net.predict(board)
            v = v[0]
            max_ucb = -1000
            action = -1
            for i in range(pi.shape[0]):
                if((display,i) not in self.Nsa):
                    self.Nsa[(display,i)] = 0
                self.Nsa[(display,i)] += 1
                n_state,cuurent_player = self.game.getNextState(board,player,i)
                _,v = self.net.predict(n_state)
                v = v[0]
                ucb = v + 2*(np.log(self.Ns[display])/self.Nsa[(display,i)])**0.5
                if(ucb>max_ucb and valid_move[i] == 1):
                    max_ucb = ucb
                    action = i
            #print("max_ucb=",max_ucb)
            #print("action = ",action)
            next_state,current_player = self.game.getNextState(board,player,action)
            player = current_player 
            board = self.game.getCanonicalForm(next_state,player)
            v = self.game.getGameEnded(board,player)
        return -v
