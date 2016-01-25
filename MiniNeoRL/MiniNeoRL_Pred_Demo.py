from neo.Agent import Agent
from neo.Hierarchy import Hierarchy
import numpy as np

sequence = [
        [ 0.0, 0.0, 0.0, 1.0 ],
        [ 0.0, 0.0, 1.0, 1.0 ],
        [ 0.0, 1.0, 0.0, 0.0 ],
        [ 0.0, 1.0, 0.0, 0.0 ],
        [ 0.0, 1.0, 0.0, 0.0 ],
        [ 0.0, 1.0, 0.0, 0.0 ],
        [ 1.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 1.0, 1.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 1.0 ],
        [ 1.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 1.0, 0.0, 0.0 ],
        [ 1.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 1.0, 1.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 1.0 ],
        [ 1.0, 0.0, 0.0, 0.0 ],
        [ 1.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 1.0, 0.0, 0.0 ],
        [ 1.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 1.0, 1.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 1.0 ],
        [ 1.0, 0.0, 0.0, 0.0 ],
        [ 0.0, 1.0, 0.0, 0.0 ]
    ]

h = Hierarchy(4, [ 40 ], -0.01, 0.01, 0.1)

averageError = 0

for i in range(0, 10000):
    h.simStep(np.matrix([sequence[i % len(sequence)]]).T, 0.01, 0.1, 0.00001, 0.95)

    error = None

    if np.allclose(np.greater(h.getPrediction(), 0.5), np.matrix([sequence[(i + 1) % len(sequence)]]).T):
        error = 0
    else:
        error = 1

    averageError = 0.99 * averageError + 0.01 * error

    print(h._layers[0]._states)

    print(str(i % 4) + str(np.matrix([sequence[i % len(sequence)]]).T.ravel()) + " " + str(np.greater(h.getPrediction(), 0.5).ravel()) + " Error: " + str(error) + " Average Error: " + str(averageError))