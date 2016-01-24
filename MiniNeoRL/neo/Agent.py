import numpy as np
from neo.Layer import Layer

class Agent:
    """A hierarchy of fully connected NeoRL layers that functions as a reinforcement learning agent"""

    def __init__(self, numInputs, numActions, layerSizes, initMinWeight, initMaxWeight, activeRatio):
        self._layers = []

        self._numInputs = numInputs
        self._numActions = numActions

        self._actions = np.zeros((numActions, 1))
        self._actionsExploratory = np.zeros((numActions, 1))

        self._exploration = np.zeros((numActions, 1))

        # Create layers
        for l in range(0, len(layerSizes)):
            layer = None

            if l == 0:
                if l < len(layerSizes) - 1:
                    layer = Layer(numInputs + numActions + 3, layerSizes[l], layerSizes[l + 1], initMinWeight, initMaxWeight, activeRatio)
                else:
                    layer = Layer(numInputs + numActions + 3, layerSizes[l], 1, initMinWeight, initMaxWeight, activeRatio)
            else:
                if l < len(layerSizes) - 1:
                    layer = Layer(layerSizes[l - 1], layerSizes[l], layerSizes[l + 1], initMinWeight, initMaxWeight, activeRatio)
                else:
                    layer = Layer(layerSizes[l - 1], layerSizes[l], 1, initMinWeight, initMaxWeight, activeRatio)

            self._layers.append(layer)

    def simStep(self, reward, exploration, explorationDecay, input, learnEncoderRate, learnDecoderRate, learnBiasRate, traceDecay, thetaMin, thetaMax, LViAlpha, averageAbsTDErrorDecay):
        assert(len(input) == self._numInputs)

        usedInputArr = []

        for i in range(0, self._numInputs):
            usedInputArr.append(input.item(i))

        for i in range(0, self._numActions):
            usedInputArr.append(self._actionsExploratory.item(i))

        # Placeholders for PVi, LVe, and LVi
        usedInputArr.append(0.0)
        usedInputArr.append(0.0)
        usedInputArr.append(0.0)

        usedInput = np.matrix([ usedInputArr ]).T

        # Up pass
        for l in range(0, len(self._layers)):
            if l == 0:
                self._layers[l].upPass(usedInput)
            else:
                self._layers[l].upPass(self._layers[l - 1]._states)

        # Down pass
        for l in range(0, len(self._layers)):
            rl = len(self._layers) - 1 - l

            if rl < len(self._layers) - 1:
                self._layers[rl].downPass(self._layers[rl + 1]._predictions, rl != 0)
            else:
                self._layers[rl].downPass(np.matrix([[ 0 ]]), rl != 0)

        # Get PVi, LVe, LVi
        PVi = self.getPrediction().item(self._numInputs + self._numActions + 0)
        LVe = self.getPrediction().item(self._numInputs + self._numActions + 1)
        LVi = self.getPrediction().item(self._numInputs + self._numActions + 2)

        PVfilter = PVi < thetaMin or reward < thetaMin or PVi > thetaMax or reward > thetaMax

        tdError = 0.0

        if PVfilter:
            tdError = reward - PVi
        else:
            tdError = LVe - LVi

        # Update Q weights
        predInputArr = []

        for i in range(0, self._numInputs):
            predInputArr.append(input.item(i))

        if tdError > 0.0:
            for i in range(0, self._numActions):
                predInputArr.append(self._actionsExploratory.item(i))
        else:
            for i in range(0, self._numActions):
                predInputArr.append(self._actions.item(i))

        # Learn new PVi, LVe, LVi values
        predInputArr.append(reward)
 
        if PVfilter:
            predInputArr.append(reward)
        else:
            predInputArr.append(LVe)

        predInputArr.append(LViAlpha * (LVe - LVi) + LVi)

        predInput = np.matrix([ predInputArr ]).T

        # Learn
        for l in range(0, len(self._layers)):
            if l == 0:
                if l < len(self._layers) - 1:
                    self._layers[l].learn(predInput, self._layers[l + 1]._predictionsPrev, learnEncoderRate, learnDecoderRate, learnBiasRate, traceDecay)
                else:
                    self._layers[l].learn(predInput, np.matrix([[ 0 ]]), learnEncoderRate, learnDecoderRate, learnBiasRate, traceDecay)
            else:
                if l < len(self._layers) - 1:
                    self._layers[l].learn(self._layers[l - 1]._states, self._layers[l + 1]._predictionsPrev, learnEncoderRate, learnDecoderRate, learnBiasRate, traceDecay)
                else:
                    self._layers[l].learn(self._layers[l - 1]._states, np.matrix([[ 0 ]]), learnEncoderRate, learnDecoderRate, learnBiasRate, traceDecay)

        # Determine action
        self._exploration = (1.0 - explorationDecay) * self._exploration + explorationDecay * np.random.normal() * exploration

        for i in range(0, self._numActions):
            self._actions[i] = np.minimum(1.0, np.maximum(-1.0, self.getPrediction().item(self._numInputs + i)))

            self._actionsExploratory[i] = np.minimum(1.0, np.maximum(-1.0, self._actions[i] + self._exploration))
            
            #if np.random.rand() < exploration:
            #    self._actionsExploratory[i] = np.random.rand() * 2.0 - 1.0
            #else:
            #    self._actionsExploratory[i] = self._actions[i]

    def getPrediction(self):
        return self._layers[0]._predictions

    def getActions(self):
        return self._actionsExploratory
          
       