###
# PlayerAI_3.py
# using helper files from week 4 of Columbia edx AI course, plays 2048 poorly lol
###

from random import randint
from BaseAI_3 import BaseAI
import heapq
import math
import numpy as np
import time


class PlayerAI(BaseAI):
    def __init__(self, weight):
        self.weight = weight

    # Evaluation function
    def eval(self, grid):
        #return self.weights[0]*self.grad(grid) + self.weights[1]*self.smooth(grid) + self.weights[2]*math.log2(len(grid.getAvailableCells())) + self.weights[3]*grid.getMaxTile()
        return self.weight[1]*math.log2(len(grid.getAvailableCells())) + self.weight[2]*self.grad(grid) - self.weight[3]*self.smooth(grid) + self.weight[0]*grid.getMaxTile() + self.weight[4]*self.cornerScore(grid)


    # function being minimised in minimax
    def minimise(self, grid, alpha, beta, depth, t, prev):
        ## returns tuple of (move, utility)
        if not grid.canMove() or depth == 0 or time.clock() - t > 0.15:
            if not grid.canMove():
                return (prev, 0)
            return (prev, self.eval(grid))

        (minChild, minEval) = (None, float('inf'))


        for a in self.queueMoves(grid):
            newGrid = grid.clone()
            newGrid.move(a[1])
            eval = self.maximise(newGrid, alpha, beta, depth - 1, t, a[1])[1]

            if minEval > eval:
                (minChild, minEval) = (a[1], eval)

            if minEval <= alpha:
                break

            if minEval < beta:
                beta = minEval

        return (minChild, minEval)

    # function being maximised in minimax
    def maximise(self, grid, alpha, beta, depth, t, prev):
        ## returns tuple of (move, utility)
        if (not grid.canMove()) or depth == 0 or time.clock() - t > 0.15:
            if not grid.canMove():
                return (prev, 0)
            return (prev, self.eval(grid))

        (maxChild, maxEval) = (None, float('-inf'))

        for a in self.queueMoves(grid):
            newGrid = grid.clone()
            newGrid.move(a[1])
            eval = self.minimise(newGrid, alpha, beta, depth - 1, t, a[1])[1]

            if maxEval < eval:
                (maxChild, maxEval) = (a[1], eval)

            if maxEval >= beta:
                break

            if maxEval > alpha:
                alpha = maxEval

        return (maxChild, maxEval)

    def getMove(self, grid):
        ## returns move
        newGrid = grid.clone()
        start = time.clock()
        (mv, util) = self.maximise(newGrid, float('-inf'), float('inf'), 50, start, None)
        return mv

    # finds to what extent tiles are monotonically increasing in some direction
    def grad(self, grid):
        linesx =[[],[],[],[]]
        linesy = [[],[],[],[]]
        f = lambda line: np.diff(line)
        for i in range(4):
            for j in range(4):
                linesx[i].append(grid.getCellValue([i,j]))
                linesy[i].append(grid.getCellValue([j,i]))

        dx = list(map(f, linesx))
        dy = list(map(f, linesy))
        for i in range(0,4):
            truth = np.all([(max(dx[i]) <= 0 or min(dx[i]) >= 0) and (max(dy[i]) <= 0 or min(dy[i]) >= 0)])

        return int(truth)

    # finds to what extent tiles are similar to thier neighbours
    def smooth(self, grid):
        penalty = 0

        for i in range(4):
            for j in range(4):
                try:
                    penalty += abs(grid.getCellValue([i,j]) - grid.getCellValue([i, j-1]))
                except (IndexError, TypeError):
                    pass
                try:
                    penalty += abs(grid.getCellValue([i,j]) - grid.getCellValue([i, j+1]))
                except (IndexError, TypeError):
                    pass
                try:
                    penalty += abs(grid.getCellValue([i,j]) - grid.getCellValue([i+1, j]))
                except (IndexError, TypeError):
                    pass
                try:
                    penalty += abs(grid.getCellValue([i,j]) - grid.getCellValue([i-1, j]))
                except (IndexError, TypeError):
                    pass


        return penalty/4
    
    def queueMoves(self, grid):
        h = []
        for a in grid.getAvailableMoves():
            newGrid = grid.clone()
            newGrid.move(a)
            heapq.heappush(h, (len(grid.getAvailableCells()), a))

        return [heapq.heappop(h) for i in range(len(h))]

    # finds if one of the corner tiles is the max tile
    def cornerScore(self, grid):
        if grid.getCellValue([0,0]) == grid.getMaxTile() or grid.getCellValue([0,3]) == grid.getMaxTile() or grid.getCellValue([3,0]) == grid.getMaxTile() or grid.getCellValue([3,3]) == grid.getMaxTile():
            return 1
        else:
            return 0
