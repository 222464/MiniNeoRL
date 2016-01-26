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

        self._qPredictiveWeights = np.random.rand(1, layerSizes[0])
        self._qPredictiveTraces = np.zeros((1, layerSizes[0]))

        self._qFeedBackWeights = np.random.rand(1, layerSizes[0])
        self._qFeedBackTraces = np.zeros((1, layerSizes[0]))

        self._averageAbsTDError = 1.0

        self._prevValue = 0.0

        # Create layers
        for l in range(0, len(layerSizes)):
            layer = None

            if l == 0:
                if l < len(layerSizes) - 1:
                    layer = Layer(numInputs + numActions, layerSizes[l], layerSizes[l + 1], initMinWeight, initMaxWeight, activeRatio)
                else:
                    layer = Layer(numInputs + numActions, layerSizes[l], 1, initMinWeight, initMaxWeight, activeRatio)
            else:
                if l < len(layerSizes) - 1:
                    layer = Layer(layerSizes[l - 1], layerSizes[l], layerSizes[l + 1], initMinWeight, initMaxWeight, activeRatio)
                else:
                    layer = Layer(layerSizes[l - 1], layerSizes[l], 1, initMinWeight, initMaxWeight, activeRatio)

            self._layers.append(layer)

    def simStep(self, reward, qAlpha, qGamma, exploration, explorationDecay, input, learnEncoderRate, learnDecoderRate, learnBiasRate, traceDecay, reinforceBias, averageAbsTDErrorDecay):
        assert(len(input) == self._numInputs)

        usedInputArr = []

        for i in range(0, self._numInputs):
            usedInputArr.append(input.item(i))

        for i in range(0, self._numActions):
            usedInputArr.append(self._actionsExploratory.item(i))

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

        # Get Q
        q = 0.0

        if len(self._layers) > 1:
            q = np.dot(self._qPredictiveWeights, self._layers[0]._states) + np.dot(self._qFeedBackWeights, self._layers[1]._predictions)
        else:
            q = np.dot(self._qPredictiveWeights, self._layers[0]._states)

        tdError = reward + qGamma * q.item(0) - self._prevValue

        self._qPredictiveWeights += qAlpha * tdError * self._qPredictiveTraces

        self._qPredictiveTraces = self._qPredictiveTraces * traceDecay + self._layers[0]._states.T

        if len(self._layers) > 1:
            self._qFeedBackWeights += qAlpha * tdError * self._qFeedBackTraces

            self._qFeedBackTraces = self._qFeedBackTraces * traceDecay + self._layers[1]._predictions.T

        predInputArr = []

        for i in range(0, self._numInputs):
            predInputArr.append(input.item(i))

        for i in range(0, self._numActions):
            predInputArr.append(self._actionsExploratory.item(i))

        predInput = np.matrix([ predInputArr ]).T

        reinforce = tdError / np.maximum(0.0001, self._averageAbsTDError) + reinforceBias

        self._averageAbsTDError = (1.0 - averageAbsTDErrorDecay) * self._averageAbsTDError + averageAbsTDErrorDecay * np.absolute(tdError)

        # Learn
        for l in range(0, len(self._layers)):
            if l == 0:
                if l < len(self._layers) - 1:
                    self._layers[l].learnRL(reinforce, predInput, self._layers[l + 1]._predictionsPrev, learnEncoderRate, learnDecoderRate, learnBiasRate, traceDecay)
                else:
                    self._layers[l].learnRL(reinforce, predInput, np.matrix([[ 0 ]]), learnEncoderRate, learnDecoderRate, learnBiasRate, traceDecay)
            else:
                if l < len(self._layers) - 1:
                    self._layers[l].learnRL(reinforce, self._layers[l - 1]._states, self._layers[l + 1]._predictionsPrev, learnEncoderRate, learnDecoderRate, learnBiasRate, traceDecay)
                else:
                    self._layers[l].learnRL(reinforce, self._layers[l - 1]._states, np.matrix([[ 0 ]]), learnEncoderRate, learnDecoderRate, learnBiasRate, traceDecay)

        # Determine action
        self._exploration = (1.0 - explorationDecay) * self._exploration + explorationDecay * np.random.normal() * exploration

        for i in range(0, self._numActions):
            self._actions[i] = np.minimum(1.0, np.maximum(-1.0, self.getPrediction().item(self._numInputs + i)))

            self._actionsExploratory[i] = np.minimum(1.0, np.maximum(-1.0, self._actions[i] + self._exploration.item(i)))

        self._prevValue = q.item(0)

    def getPrediction(self):
        return self._layers[0]._predictions

    def getActions(self):
        return self._actionsExploratory
          
       