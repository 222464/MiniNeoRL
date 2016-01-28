import numpy as np
from neo.Layer import Layer

class Hierarchy:
    """A hierarchy of fully connected NeoRL layers"""

    def __init__(self, numInputs, layerSizes, initMinWeight, initMaxWeight, activeRatio):
        self._layers = []

        # Create layers
        for l in range(0, len(layerSizes)):
            layer = None

            if l == 0:
                if l < len(layerSizes) - 1:
                    layer = Layer(numInputs, layerSizes[l], layerSizes[l], initMinWeight, initMaxWeight, activeRatio)
                else:
                    layer = Layer(numInputs, layerSizes[l], 1, initMinWeight, initMaxWeight, activeRatio)
            else:
                if l < len(layerSizes) - 1:
                    layer = Layer(layerSizes[l - 1], layerSizes[l], layerSizes[l], initMinWeight, initMaxWeight, activeRatio)
                else:
                    layer = Layer(layerSizes[l - 1], layerSizes[l], 1, initMinWeight, initMaxWeight, activeRatio)

            self._layers.append(layer)

    def simStep(self, input, learnEncoderRate, learnRecurrentRate, learnDecoderRate, learnBiasRate, traceDecay):
        # Up pass
        for l in range(0, len(self._layers)):
            if l == 0:
                self._layers[l].upPass(input)
            else:
                self._layers[l].upPass(self._layers[l - 1]._states)

        # Down pass
        for l in range(0, len(self._layers)):
            rl = len(self._layers) - 1 - l

            if rl < len(self._layers) - 1:
                self._layers[rl].downPass(self._layers[rl + 1]._predictions, rl != 0)
            else:
                self._layers[rl].downPass(np.matrix([[ 0 ]]), rl != 0)

        # Learn
        for l in range(0, len(self._layers)):
            if l == 0:
                if l < len(self._layers) - 1:
                    self._layers[l].learn(input, self._layers[l + 1]._predictionsPrev, learnEncoderRate, learnRecurrentRate, learnDecoderRate, learnBiasRate, traceDecay)
                else:
                    self._layers[l].learn(input, np.matrix([[ 0 ]]), learnEncoderRate, learnRecurrentRate, learnDecoderRate, learnBiasRate, traceDecay)
            else:
                if l < len(self._layers) - 1:
                    self._layers[l].learn(self._layers[l - 1]._states, self._layers[l + 1]._predictionsPrev, learnEncoderRate, learnRecurrentRate, learnDecoderRate, learnBiasRate, traceDecay)
                else:
                    self._layers[l].learn(self._layers[l - 1]._states, np.matrix([[ 0 ]]), learnEncoderRate, learnRecurrentRate, learnDecoderRate, learnBiasRate, traceDecay)

    def getPrediction(self):
        return self._layers[0]._predictions
          
       