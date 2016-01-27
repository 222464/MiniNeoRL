from neo.Agent import Agent
from neo.Hierarchy import Hierarchy
import numpy as np
import pygame

# The environment
displayWidth = 600
displayHeight = 600

ballPosition = np.array([ np.random.rand(), np.random.rand() * 0.5 + 0.5 ])

ballVelocity = np.array([ 0.353, 1.0 ]) * 0.04

paddleX = 0.5

ballRadius = 16.0 / displayWidth
paddleRadius = 64.0 / displayWidth

encoderSize = 10
numInputs = 5
numActions = 1

a = Agent(numInputs * encoderSize, numActions, [ 200 ], -0.1, 0.1, 0.04)

averageReward = 0.0

reward = 0.0
prevReward = 0.0

rewardPunishmentTime = 2.0
punishmentTimer = 0.0
rewardTimer = 0.0

# Resources
ballImage = pygame.image.load("ball.png")
paddleImage = pygame.image.load("paddle.png")

# Game setup
pygame.init()

display = pygame.display.set_mode((displayWidth, displayHeight))
clock = pygame.time.Clock()
done = False

dir = 1.0

timer = 0.0

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                dir = -1.0
            if event.key == pygame.K_RIGHT:
                dir = 1.0

    # Update physics
    ballPosition += ballVelocity
    #timer += 1.0
    #ballPosition[0] = np.sin(timer * 0.05) * 0.5 + 0.5
    #ballPosition[1] = 0.5

    if ballPosition[0] < 0.0:
        ballPosition[0] = 0.0
        ballVelocity[0] *= -1.0
    elif ballPosition[0] > 1.0:
        ballPosition[0] = 1.0
        ballVelocity[0] *= -1.0

    if ballPosition[1] < 32.0 / displayWidth:       
        # If hit paddle
        if ballPosition[0] + ballRadius > paddleX - paddleRadius and ballPosition[0] - ballRadius < paddleX + paddleRadius:
            rewardTimer = rewardPunishmentTime

            # Bounce ball
            ballPosition[1] = 32.0 / displayWidth
            ballVelocity[1] *= -1.0
        else:
            punishmentTimer = rewardPunishmentTime

            # Reset ball
            ballPosition = np.array([ np.random.rand(), np.random.rand() * 0.5 + 0.5 ])

            if np.random.rand() < 0.5:
                ballVelocity = np.array([ 0.353, 1.0 ]) * 0.04
            else:
                ballVelocity = np.array([ -0.353, 1.0 ]) * 0.04

    elif ballPosition[1] > 1.0:
        ballPosition[1] = 1.0
        ballVelocity[1] *= -1.0

    reward = (rewardTimer > 0.0)# - (punishmentTimer > 0.0)

    #reward = reward * 0.5 + 0.5

    averageReward = 0.99 * averageReward + 0.01 * reward

    # Control
    inputs = [ paddleX * 2.0 - 1.0, ballPosition[0] * 2.0 - 1.0, ballPosition[1] * 2.0 - 1.0, ballVelocity[0] * 30.0, ballVelocity[1] * 30.0 ]

    assert(len(inputs) == numInputs)

    inputArr = []

    encoderSharpness = 30.0

    for v in inputs:
        for i in range(0, encoderSize):
            center = i / encoderSize * 2.0 - 1.0
            delta = center - v
            #intensity = np.exp(-delta * delta * encoderSharpness)

            intensity = np.absolute(delta) < 0.5 / encoderSize

            inputArr.append(intensity)

    reward = dir * paddleX * 0.01

    #reward = np.abs(paddleX - ballPosition[0]) < 0.1

    a.simStep(reward, 0.002, 0.95, 0.12, 1.0, np.matrix([inputArr]).T, 0.01, 0.01, 0.0005, 0.92)

    print(a.getActions())

    prevReward = reward

    if rewardTimer > 0.0:
        rewardTimer -= 1.0
    if punishmentTimer > 0.0:
        punishmentTimer -= 1.0

    paddleX = np.minimum(1.0, np.maximum(0.0, paddleX + 0.2 * np.sum(a.getActions()) / numActions))

    # Render
    display.fill((255,255,255))

    display.blit(paddleImage, (displayWidth * paddleX - 64.0, displayHeight - 32.0))
    display.blit(ballImage, (displayWidth * ballPosition[0] - 16.0, displayHeight * (1.0 - ballPosition[1]) - 16.0))

    pygame.display.flip()
    clock.tick(60)
