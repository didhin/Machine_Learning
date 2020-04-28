from gym import spaces
import numpy as np
import random
import pickle

class TicTacToe:
    def __init__(self):
        self.state = [0]*9
        self.player1 = None 
        self.player2 = None


    def evaluate(self):
        for i in range(3):
            if (self.state[i * 3] + self.state[i * 3 + 1] + self.state[i * 3 + 2]) == 15:
                return 1.0, True
        
        for i in range(3):
            if (self.state[i + 0] + self.state[i + 3] + self.state[i + 6]) == 15:
                return 1.0, True
        
        if (self.state[0] + self.state[4] + self.state[8]) == 15:
            return 1.0, True
        if (self.state[2] + self.state[4] + self.state[6]) == 15:
            return 1.0, True

        if not any(space == 0 for space in self.state):
            return 0.0, True

        return 0.0, False
    
    def step(self, player_mode, move):
        self.state[move-1]= self.nextMove(player_mode)
        reward, done = self.evaluate()
        return reward, done

    def possibleMoves(self):
        empty_spots =  [moves + 1 for moves, spot in enumerate(self.state) if spot == 0]
        return empty_spots

    def nextMove(self, player_mode):
       
        if(player_mode):
            self.player1.options = random.sample(self.player1.options, len(self.player1.options))
            return self.player1.options.pop()
   
        else:
            self.player2.options = random.sample(self.player2.options, len(self.player2.options))
            return self.player2.options.pop()


    def beginTraining(self, oddp, evenp, iterations, odd=True, verbose = False):
        self.player1=oddp
        self.player2=evenp

        print ("Training Started")
        for i in range(iterations):
            if verbose: print("training ", i)
            self.player1.game_begin()
            self.player2.game_begin()
            self.reset()
            done = False

            player_mode = odd
            while not done:
                if player_mode:
                    move = self.player1.epslion_greedy(self.state, self.possibleMoves())
                else:
                    move = self.player2.epslion_greedy(self.state, self.possibleMoves())

                
                reward, done = self.step(player_mode, move)

                if (reward == 1):  
                    if (player_mode):
                        self.player1.updateQ(10, self.state, self.possibleMoves())
                        self.player2.updateQ(-10, self.state, self.possibleMoves())
                    else:
                        self.player1.updateQ(-10, self.state, self.possibleMoves())
                        self.player2.updateQ(10, self.state, self.possibleMoves())
                elif (done == False):
                    if (player_mode):
                        self.player1.updateQ(-1, self.state, self.possibleMoves())

                else:
                    self.player1.updateQ(reward, self.state, self.possibleMoves())
                    self.player2.updateQ(reward, self.state, self.possibleMoves())
                    
                

                player_mode = not player_mode 
        print ("training has been Completed")

    #saving the Q-tables
    def saveStates(self):
        self.player1.saveQ("oddPolicy")
        self.player2.saveQ("evenPolicy")
        
        

    def getQ(self):
        return self.player1.Q_dict, self.player2.Q_dict

    def reset(self):
        self.state = [0] * 9

