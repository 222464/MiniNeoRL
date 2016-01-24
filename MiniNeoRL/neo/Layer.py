import numpy as np
from operator import itemgetter, attrgetter, methodcaller

class Layer:
    """A fully-connected NeoRL layer"""

    def __init__(self, numInputs, numHidden, numFeedBack, initMinWeight, initMaxWeight, activeRatio):
        self._input = np.zeros((numInputs, 1))

        self._feedForwardWeights = np.random.rand(numHidden, numInputs) * (initMaxWeight - initMinWeight) + initMinWeight
        self._feedForwardTraces = np.zeros((numHidden, numInputs))

        self._recurrentWeights = np.random.rand(numHidden, numHidden) * (initMaxWeight - initMinWeight) + initMinWeight
        self._recurrentTraces = np.zeros((numHidden, numHidden))

        self._predictiveWeights = np.random.rand(numInputs, numHidden) * (initMaxWeight - initMinWeight) + initMinWeight
        self._predictiveTraces = np.zeros((numInputs, numHidden))
 
        self._feedBackWeights = np.random.rand(numInputs, numFeedBack) * (initMaxWeight - initMinWeight) + initMinWeight
        self._feedBackTraces = np.zeros((numInputs, numFeedBack))
 
        self._biases = np.random.rand(numHidden, 1) * (initMaxWeight - initMinWeight) + initMinWeight

        self._statesRecurrent = np.zeros((numHidden, 1))
        self._statesFeedForward = np.zeros((numHidden, 1))

        self._states = np.zeros((numHidden, 1))
        self._statesPrev = np.zeros((numHidden, 1))

        self._feedForwardLearn = np.zeros((numHidden, 1))
        self._recurrentLearn = np.zeros((numHidden, 1))

        self._predictions = np.zeros((numInputs, 1))
        self._predictionsPrev = np.zeros((numInputs, 1))

        self._activeRatio = activeRatio

    def upPass(self, input):
        self._input = input
        self._statesPrev = self._states

        numActive = int(self._activeRatio * len(self._states))
  
        # Activate
        recurrentActivations = np.dot(self._recurrentWeights, self._statesPrev)
        feedForwardActivations = self._biases + np.dot(self._feedForwardWeights, input)
       
        # Generate tuples for sorting
        recurrentActivationsPairs = []
        feedForwardActivationsPairs = []

        for i in range(0, len(self._states)):
            recurrentActivationsPairs.append((recurrentActivations[i], i))
            feedForwardActivationsPairs.append((feedForwardActivations[i], i))

        # Sort
        recurrentActivationsPairs = sorted(recurrentActivationsPairs, key=itemgetter(0))
        feedForwardActivationsPairs = sorted(feedForwardActivationsPairs, key=itemgetter(0))

        # Use sorted information for inhibition
        self._statesRecurrent = np.zeros((len(self._states), 1))
        self._statesFeedForward = np.zeros((len(self._states), 1))

        for i in range(0, numActive):
            self._statesRecurrent[recurrentActivationsPairs[len(recurrentActivationsPairs) - 1 - i][1]] = 1.0
            self._statesFeedForward[feedForwardActivationsPairs[len(feedForwardActivationsPairs) - 1 - i][1]] = 1.0

        self._states = np.maximum(self._statesFeedForward, self._statesRecurrent)

    def downPass(self, feedBack, thresholdedPred = True):
        self._predictionsPrev = self._predictions

        # Find states
        self._predictions = np.dot(self._predictiveWeights, self._states) + np.dot(self._feedBackWeights, feedBack)

        if thresholdedPred:
            self._predictions[self._predictions > 0.5] = 1.0
            self._predictions[self._predictions <= 0.5] = 0.0
        
    def learn(self, target, feedBackPrev, learnEncoderRate, learnDecoderRate, learnBiasRate, traceDecay):
        # Find prediction error
        predError = target - self._predictionsPrev

        # Propagate error
        hiddenError = np.dot(self._predictiveWeights.T, predError)

        hiddenError = np.multiply(hiddenError, self._statesPrev)

        # Update feed forward and recurrent weights
        self._feedForwardWeights += learnEncoderRate * np.dot(hiddenError.T, self._feedForwardTraces)
        self._recurrentWeights += learnEncoderRate * np.dot(hiddenError.T, self._recurrentTraces)

        self._feedForwardTraces = self._feedForwardTraces * traceDecay + np.repeat(self._input.T, len(self._states), 0)
        self._recurrentTraces = self._recurrentTraces * traceDecay + np.dot(self._statesFeedForward - self._statesRecurrent, self._statesPrev.T)

        # Update predictive and feed back weights
        self._predictiveWeights += learnDecoderRate * np.dot(predError, self._statesPrev.T)
        self._feedBackWeights += learnDecoderRate * np.dot(predError, feedBackPrev.T)

        # Update thresholds
        self._biases += learnBiasRate * np.dot(self._predictiveWeights.T, predError)

    def learnRL(self, reward, target, feedBackPrev, learnEncoderRate, learnDecoderRate, learnBiasRate, traceDecay):
        # Find prediction error
        predError = target - self._predictionsPrev

        # Propagate error
        hiddenError = np.dot(self._predictiveWeights.T, predError)

        hiddenError = np.multiply(hiddenError, self._statesPrev)

        # Update feed forward and recurrent weights
        self._feedForwardWeights += learnEncoderRate * np.dot(hiddenError.T, self._feedForwardTraces)
        self._recurrentWeights += learnEncoderRate * np.dot(hiddenError.T, self._recurrentTraces)

        self._feedForwardTraces = self._feedForwardTraces * traceDecay + np.repeat(self._input.T, len(self._states), 0)
        self._recurrentTraces = self._recurrentTraces * traceDecay + np.dot(self._statesFeedForward - self._statesRecurrent, self._statesPrev.T)

        # Update predictive and feed back weights
        self._predictiveTraces = self._predictiveTraces * traceDecay + np.dot(predError, self._statesPrev.T)
        self._feedBackTraces = self._feedBackTraces * traceDecay + np.dot(predError, feedBackPrev.T)

        self._predictiveWeights += learnDecoderRate * reward * self._predictiveTraces
        self._feedBackWeights += learnDecoderRate * reward * self._feedBackTraces

        # Update thresholds
        self._biases += learnBiasRate * np.dot(self._predictiveWeights.T, predError)