import numpy as np
from models import PPOAgent
from utils import *
import pickle

import wandb

WANDB_MODE = 'online'
WANDB_API_KEY = '83230c40e1c562f3ef56bf082e31911eaaad4ed9'


class PPOMultiAgentTrainer:
    # def __init__(self, env, neuralNetSpecs, learningRate, modelPath=None):
    def __init__(self, env, **kwargs):

        self.env = env
        self.numAgents = env.numAgents
        self.actionDim = env.action_space.n

        if 'modelPath' in kwargs:
            modelPath = kwargs.get('modelPath')
            with open(modelPath, 'rb') as fp:
                self.agents = pickle.load(fp)
                self.stateSpaceDim = np.prod(env.observation_space.shape)
        else:
            assert 'neuralNetSpecs' in kwargs
            neuralNetSpecs = kwargs.get('neuralNetSpecs')
            if 'learningRate' in kwargs:
                learningRate = kwargs.get('learningRate')
            else:
                learningRate = 1e-4
            if isinstance(neuralNetSpecs[0], ProtoConvNet):
                self.stateSpaceDim = env.observation_space.shape
            else:
                self.stateSpaceDim = np.prod(env.observation_space.shape)
            self.isRecurrent = any(isinstance(net, ProtoLSTMNet) for net in neuralNetSpecs)
            self.agents = []
            for i in range(self.numAgents):
                self.agents.append(PPOAgent(self.stateSpaceDim, self.actionDim, neuralNetSpecs, learningRate))

        self.totalSteps = 0
        self.rewardsHistory = []

        wandb.init(project="CommonsGamesTesting",
                   name="TestRun",
                   notes="Standard Config - just trying to get it running",
                   mode=WANDB_MODE)

        wandb.watch()

    def interact(self, observations, rnnState, training=False):
        self.totalSteps += 1
        actions = [{} for _ in range(self.numAgents)]
        for k in range(self.numAgents):
            if observations[k] is not None:
                observation = np.reshape(observations[k], newshape=(-1, self.stateSpaceDim)).astype(np.float32)
                actions[k], rnnState[k] = self.agents[k].act(observation, rnnState[k][0], rnnState[k][1])
        newObservations, rewards, done, _ = self.env.step(actions)

        if not training:
            self.env.render()
        done = np.logical_or.reduce(done)
        return newObservations, rewards, done

    def test(self, interactionLength):
        observations = self.env.reset()
        rnnState = []
        for k in range(self.numAgents):
            rnnState.append((np.zeros((1, 1, 128)), np.zeros((1, 1, 128))))
        for t in range(interactionLength):
            newObservations, rewards, done = self.interact(observations, rnnState)
            if done:
                break
            observations = newObservations

    def train(self, maxEpisodes, maxEpisodeLength, logPeriod, savePeriod, savePath):
        cumRewards = np.zeros((self.numAgents,))
        avgEpisodeLength = 0
        for episode in range(1, maxEpisodes + 1):
            observations = self.env.reset()
            rnnState = []
            for k in range(self.numAgents):
                rnnState.append((np.zeros((1, 1, 128)), np.zeros((1, 1, 128))))
            for t in range(maxEpisodeLength):
                newObservations, rewards, done = self.interact(observations, rnnState, training=True)
                for k in range(self.numAgents):
                    if observations[k] is not None:
                        self.agents[k].memory['rewards'].append(rewards[k])
                        self.agents[k].memory['terminalFlags'].append(done)

                cumRewards += rewards
                if done:
                    break
                observations = newObservations

            avgEpisodeLength += t

            self.learn()

            if episode % logPeriod == 0:
                avgEpisodeLength = int(avgEpisodeLength / logPeriod)
                cumRewards = (cumRewards / logPeriod)
                print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avgEpisodeLength, cumRewards))
                cumRewards = np.zeros((self.numAgents,))
                avgEpisodeLength = 0

            if episode % savePeriod == 0:
                print("Saved model")
                with open(savePath, 'wb') as fp:
                    pickle.dump(self.agents, fp)

    def learn(self):
        for k in range(self.numAgents):
            self.agents[k].learn(numEpochs=4)
            self.agents[k].clearMemory()


class PPOTrainer:
    def __init__(self, env, neuralNetSpecs, learningRate):
        self.env = env
        self.actionDim = env.action_space.n
        self.stateSpaceDim = np.prod(env.observation_space.shape)
        self.agent = PPOAgent(self.stateSpaceDim, self.actionDim, neuralNetSpecs, learningRate)
        self.totalSteps = 0
        self.rewardsHistory = []

    def train(self, maxEpisodes, maxEpisodeLength, updatePeriod, logPeriod, render=False):
        cumReward = 0
        avgEpisodeLength = 0
        for episode in range(1, maxEpisodes + 1):
            observation = self.env.reset()
            for t in range(maxEpisodeLength):
                self.totalSteps += 1

                action = self.agent.act(observation)
                newObservation, reward, done, _ = self.env.step(action)

                if render:
                    self.env.render()

                self.agent.memory['rewards'].append(reward)
                self.agent.memory['terminalFlags'].append(done)

                if self.totalSteps % updatePeriod == 0:
                    self.learn()

                cumReward += reward
                if done:
                    break
                observation = newObservation

            avgEpisodeLength += t

            if episode % logPeriod == 0:
                avgEpisodeLength = int(avgEpisodeLength / logPeriod)
                cumReward = int((cumReward / logPeriod))
                print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avgEpisodeLength, cumReward))
                cumReward = 0
                avgEpisodeLength = 0

    def learn(self):
        self.agent.learn(numEpochs=4)
        self.agent.clearMemory()
