import numpy as np


class RandomPlayer:
    """a random player for the game of santorini"""
    def __init__(self, game):
        self.game = game
        self.__class__.__call__ = self.call

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board,1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a

    def call(self,board):
        return self.play(board)

class AgentPlayer:
    def __init__(self,game,model):
        self.game = game
        self.model = model
        self.__class__.__call__ = self.call
    def play(self,board):
        pi,v = self.model.predict(board)
        valids = self.game.getValidMoves(board,1)
        pi2 = []
        for i in range(len(pi)):
            if(valids[i]==0):
                pi2.append(-1000)
            else:
                pi2.append(pi[i])
        pi2 = np.array(pi2)
        return np.argmax(pi2)
    def call(self,board):
        return self.play(board)

class HumanPlayer:
    """if you want to play it yourself"""
    def __init__(self, game):
        self.game = game
        self.__class__.__call__ = self.call

    def play(self, board):
        print(board[:2])  # print the buliding and the positions of the workers
        valid = self.game.getValidMoves(board,1)

        while True:
            a = input()
            worker, walk, build, *_ = a.split(' ')
            worker = int(worker)
            ai = self.game.env.atoi[(worker, walk, build)]
            if valid[ai]:
                break
            else:
                print('Invalid')
        return ai
    def call(self,board):
        return self.play(board)