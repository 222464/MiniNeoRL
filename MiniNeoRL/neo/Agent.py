import numpy as np
from neo.Layer import Layer

class ReplaySample:
    def __init__(self, states, feedBack, action, exploratoryAction, originalQ, newQ):
        self._states = states
        self._feedBack = feedBack
        self._originalQ = originalQ
        self._q = newQ
        self._action = action
        self._exploratoryAction = exploratoryAction

class Agent:
    """A hierarchy of fully connected NeoRL layers that functions as a reinforcement learning agent"""

    def __init__(self, numInputs, numActions, layerSizes, initMinWeight, initMaxWeight, activeRatio):
        self._layers = []

        self._numInputs = numInputs
        self._numActions = numActions

        self._actions = np.zeros((numActions, 1))
        self._actionsExploratory = np.zeros((numActions, 1))

        self._exploration = np.zeros((numActions, 1))

        self._replayBuffer = []

        self._maxReplayBufferSize = 600
        self._replayIterations = 40

        self._prevValue = 0.0

        # Create layers
        for l in range(0, len(layerSizes)):
            layer = None

            if l == 0:
                if l < len(layerSizes) - 1:
                    layer = Layer(numInputs + numActions + 1, layerSizes[l], layerSizes[l + 1], initMinWeight, initMaxWeight, activeRatio)
                else:
                    layer = Layer(numInputs + numActions + 1, layerSizes[l], 1, initMinWeight, initMaxWeight, activeRatio)
            else:
                if l < len(layerSizes) - 1:
                    layer = Layer(layerSizes[l - 1], layerSizes[l], layerSizes[l + 1], initMinWeight, initMaxWeight, activeRatio)
                else:
                    layer = Layer(layerSizes[l - 1], layerSizes[l], 1, initMinWeight, initMaxWeight, activeRatio)

            self._layers.append(layer)

    def simStep(self, reward, qAlpha, qGamma, exploration, explorationDecay, input, learnEncoderRate, learnDecoderRate, learnBiasRate, traceDecay, replayAlpha):
        assert(len(input) == self._numInputs)

        usedInputArr = []

        for i in range(0, self._numInputs):
            usedInputArr.append(input.item(i))

        for i in range(0, self._numActions):
            usedInputArr.append(self._actionsExploratory.item(i))

        # Placeholder for q
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
        q = self.getPrediction().item(self._numInputs + self._numActions)

        tdError = reward + qGamma * q - self._prevValue

        newQ = self._prevValue + qAlpha * tdError

        # Update older samples
        g = qGamma

        for i in range(0, len(self._replayBuffer)):
            ri = len(self._replayBuffer) - 1 - i

            self._replayBuffer[ri]._q += qAlpha * g * tdError

            g *= qGamma

        sample = None
        
        if len(self._layers) > 1:
            sample = ReplaySample(self._layers[0]._statesPrev, self._layers[1]._predictionsPrev, self._actions, self._actionsExploratory, self._prevValue, newQ)
        else:
            sample = ReplaySample(self._layers[0]._statesPrev, np.matrix([[ 0 ]]), self._actions, self._actionsExploratory, self._prevValue, newQ)

        self._replayBuffer.append(sample)

        while len(self._replayBuffer) > self._maxReplayBufferSize:
            self._replayBuffer.pop(0)

        predInputArr = []

        for i in range(0, self._numInputs):
            predInputArr.append(input.item(i))

        if tdError > 0.0:
            for i in range(0, self._numActions):
                predInputArr.append(self._actionsExploratory.item(i))
        else:
            for i in range(0, self._numActions):
                predInputArr.append(self._actions.item(i))

        predInputArr.append(newQ)

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

            #self._actionsExploratory[i] = np.minimum(1.0, np.maximum(-1.0, self._actions[i] + self._exploration))

            if np.random.rand() < exploration:
                self._actionsExploratory[i] = np.random.rand() * 2.0 - 1.0
            else:
                self._actionsExploratory[i] = self._actions[i]

        # Experience replay
        for iter in range(0, self._replayIterations):
            replayIndex = np.random.randint(0, len(self._replayBuffer))

            replaySample = self._replayBuffer[replayIndex]

            # Activate off of old state
            activations = np.dot(self._layers[0]._predictiveWeights, replaySample._states) + np.dot(self._layers[0]._feedBackWeights, replaySample._feedBack)

            errorsArr = []

            for i in range(0, self._numInputs):
                errorsArr.append(0.0)

            if replaySample._q >= replaySample._originalQ:
                for i in range(0, self._numActions):
                    errorsArr.append(replaySample._exploratoryAction.item(i) - activations.item(self._numInputs + i))
            else:
                for i in range(0, self._numActions):
                    errorsArr.append(replaySample._action.item(i) - activations.item(self._numInputs + i))

            errorsArr.append(replaySample._q - activations.item(self._numInputs + self._numActions))

            errors = np.matrix([errorsArr]).T

            self._layers[0]._predictiveWeights += replayAlpha * np.dot(errors, replaySample._states.T)
            self._layers[0]._feedBackWeights += replayAlpha * np.dot(errors, replaySample._feedBack.T)

        self._prevValue = q

    def getPrediction(self):
        return self._layers[0]._predictions

    def getActions(self):
        return self._actionsExploratory
          
       