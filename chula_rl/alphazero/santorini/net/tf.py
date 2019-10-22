import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from chula_rl.alphazero.game import Game
from chula_rl.alphazero.net import Net


class SantoriniNet(Net):
    """
    Args:
        args: in case you want to configure the network somehow
    """
    def __init__(self, game: Game, args):
        self.args = args
        self.nnet = Backbone(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # implement this function
        # look for inspiration from Othello which is provided in full!
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)

        board = []
        for i in range(input_boards.shape[0]):
            temp = []
            for j in range(5):
                for k in range(5):
                    temp.append(input_boards[i][0][j][k]+input_boards[i][1][j][k])
            temp = np.array(temp)
            board.append(temp)
        board = np.array(board)
        print(board.shape)

        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=board,
                            y=[target_pis, target_vs],
                            batch_size=self.args['n_bs'],
                            epochs=self.args['n_ep'])


    def predict(self, board):
        """
        board: np array with board
        """
        # implement this function
        # timing
        start = time.time()
        temp = []
        for i in range(5):
            for j in range(5):
                temp.append(board[0][i][j]+board[1][i][j])
        temp = np.array(temp)
        # preparing input
        board = temp[np.newaxis, :].astype(np.float32)
        # run
        pi, v = self.nnet.model.predict(board)
        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time() - start))
        return pi[0],v[0]

    def save_checkpoint(self,
                        folder='checkpoint',
                        filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".
                  format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self,
                        folder='checkpoint',
                        filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        self.nnet.model.load_weights(filepath)


class Backbone:
    def __init__(self, game, args):
        # implement this function
        # look for inspiration from Othello which is provided in full!
        # self.model = some keras model
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.lr = args['lr']
        self.n_ch = args['n_ch']
        self.dropout = args['dropout']

        self.input_boards = Input(
            shape=(self.board_x*
                   self.board_y))  # s: batch_size x board_x x board_y

        s_fc1 = Dropout(self.dropout)(Activation('relu')(Dense(1024,
                          use_bias=False)(self.input_boards)))  # batch_size x 1024


        s_fc2 = Dropout(self.dropout)(Activation('relu')(Dense(512, use_bias=False)(s_fc1)))  # batch_size x 1024

        self.pi = Dense(self.action_size, activation='softmax',
                        name='pi')(s_fc2)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'],
            optimizer=Adam(self.lr))

