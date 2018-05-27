###
# weightTester.py
# this is supposed to use hill climbing to optimise the parameters of the eval function in PlayerAI_3.py
# but doesnt do a very good job
###
from GameManager_3 import main
import time

# initialise variables and constants for grad descent
epsilon = 0.1
currentPoint = [1.2380646832542717, 0.7995432683478958, 0.8034683059831667, 2.1327160493827164, 0.8034683059831667]
stepSize = [0.5,0.5,0.5,0.5,0.5]
acc = 1.5

candidate = [0, 1/acc, acc, -acc, -1/acc]
f = open('output.txt', 'w')
bestScore = 0
temp=[0,0,0,0,0,0,0,0]
tScore = 0
scores = [0,]
bestScore = -100

# should repeatedly runs 2048 and updates parameters to get a better score
while max(scores) < 2047:
    before = main(currentPoint)
    f.write(str(scores))
    f.write(str(max(scores)) + str(scores.index(max(scores))))
    for i in range(len(currentPoint)):
        best = -1

        for j in range(len(candidate)):
            currentPoint[i] = currentPoint[i] + stepSize[i]*candidate[j]
            for k in range(8):
                temp[k] = main(currentPoint)
            tScore = sum(temp)/len(temp)
            scores.append(tScore)
            f.write("score:" + str(tScore) + '\n' + str(temp) + '\n')
            f.write(str(currentPoint) + '\n\n')
            currentPoint[i] = currentPoint[i] - stepSize[i]*candidate[j]

            if(tScore > bestScore):
                bestScore =  tScore
                best = j
            if candidate[best] == 0:
                stepSize[i] = stepSize[i] / acc
            else:
                currentPoint[i] = currentPoint[i] + stepSize[i]*candidate[best]
                stepSize[i] = stepSize[i]*candidate[best]
