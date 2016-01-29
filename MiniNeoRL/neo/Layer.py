import numpy as np
from operator import itemgetter, attrgetter, methodcaller

class Layer:
    """A fully-connected NeoRL layer"""

    def __init__(self, numInputs, numHidden, numFeedBack, initMinWeight, initMaxWeight, activeRatio):
        self._input = np.zeros((numInputs, 1))
        self._inputPrev = np.zeros((numInputs, 1))

        self._feedForwardWeights = np.random.rand(numHidden, numInputs) * (initMaxWeight - initMinWeight) + initMinWeight
        self._feedForwardTraces = np.zeros((numHidden, numInputs))

        self._recurrentWeights = np.random.rand(numHidden, numHidden) * (initMaxWeight - initMinWeight) + initMinWeight
        self._recurrentTraces = np.zeros((numHidden, numHidden))

        self._predictiveWeights = np.random.rand(numInputs, numHidden) * (initMaxWeight - initMinWeight) + initMinWeight
  
        self._feedBackWeights = np.random.rand(numInputs, numFeedBack) * (initMaxWeight - initMinWeight) + initMinWeight

        self._biases = np.zeros((numHidden, 1))#np.random.rand(numHidden, 1) * (initMaxWeight - initMinWeight) + initMinWeight

        self._statesRecurrent = np.zeros((numHidden, 1))
        self._statesRecurrentPrev = np.zeros((numHidden, 1))

        self._statesFeedForward = np.zeros((numHidden, 1))

        self._states = np.zeros((numHidden, 1))
        self._statesPrev = np.zeros((numHidden, 1))

        self._feedForwardLearn = np.zeros((numHidden, 1))
        self._recurrentLearn = np.zeros((numHidden, 1))

        self._predictions = np.zeros((numInputs, 1))
        self._predictionsPrev = np.zeros((numInputs, 1))

        self._averageSquaredError = np.zeros((numHidden, 1))

        self._activeRatio = activeRatio

    def upPass(self, input):
        self._inputPrev = self._input

        self._input = input

        self._statesPrev = self._states

        self._statesRecurrentPrev = self._statesRecurrent

        numActive = int(self._activeRatio * len(self._states))
  
        # Activate
        feedForwardActivations = self._biases + np.dot(self._feedForwardWeights, input)
       
        # Generate tuples for sorting
        feedForwardActivationsPairs = []

        for i in range(0, len(self._states)):
            feedForwardActivationsPairs.append((feedForwardActivations[i], i))

        # Sort
        feedForwardActivationsPairs = sorted(feedForwardActivationsPairs, key=itemgetter(0))

        # Use sorted information for inhibition
        self._statesFeedForward = np.zeros((len(self._states), 1))

        for i in range(0, numActive):
            self._statesFeedForward[feedForwardActivationsPairs[len(feedForwardActivationsPairs) - 1 - i][1]] = 1.0

        self._states = np.maximum(self._statesFeedForward, self._statesRecurrentPrev)
        
        recurrentActivations = np.dot(self._recurrentWeights, self._states)

        # Generate tuples for sorting
        recurrentActivationsPairs = []
        
        for i in range(0, len(self._states)):
            recurrentActivationsPairs.append((recurrentActivations[i], i))
            
        # Sort
        recurrentActivationsPairs = sorted(recurrentActivationsPairs, key=itemgetter(0))
        
        # Use sorted information for inhibition
        self._statesRecurrent = np.zeros((len(self._states), 1))
        
        for i in range(0, numActive):
            self._statesRecurrent[recurrentActivationsPairs[len(recurrentActivationsPairs) - 1 - i][1]] = 1.0
           
    def downPass(self, feedBack, thresholdedPred = True):
        self._predictionsPrev = self._predictions

        # Find states
        self._predictions = np.dot(self._predictiveWeights, self._states) + np.dot(self._feedBackWeights, feedBack)

        if thresholdedPred:
            self._predictions[self._predictions > 0.5] = 1.0
            self._predictions[self._predictions <= 0.5] = 0.0

    def learn(self, target, feedBackPrev, learnEncoderRate, learnRecurrentRate, learnDecoderRate, learnBiasRate, traceDecay):
        # Find prediction error
        predError = target - self._predictionsPrev

        # Propagate error
        hiddenError = np.dot(self._predictiveWeights.T, predError)

        hiddenError = np.multiply(hiddenError, self._statesPrev)

        # Update feed forward and recurrent weights 
        self._feedForwardWeights += learnEncoderRate * np.dot(hiddenError.T, self._feedForwardTraces)
        self._recurrentWeights += learnRecurrentRate * np.dot(hiddenError.T, self._recurrentTraces)

        self._recurrentTraces = self._recurrentTraces * traceDecay + np.dot(self._statesFeedForward, self._statesPrev.T)
        self._feedForwardTraces = self._feedForwardTraces * traceDecay + np.dot(self._statesFeedForward, self._input.T)
        
        # Update predictive and feed back weights
        self._predictiveWeights += learnDecoderRate * np.dot(predError, self._statesPrev.T)
        self._feedBackWeights += learnDecoderRate * np.dot(predError, feedBackPrev.T)

        # Update thresholds
        self._biases += learnBiasRate * (self._activeRatio - self._statesFeedForward)