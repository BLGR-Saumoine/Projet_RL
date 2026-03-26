from collections import namedtuple, deque
import random
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'gameIsOver'))

class ReplayMemory(object):

    def __init__(self, capacity, usePer = False, maxPrio = 1.0, alphaPrio = 0.6, beta_start=0.4, beta_frames=100000):
      self.buffer = deque([], maxlen=capacity)
      self.usePer = usePer
      if usePer :
        self.priority = deque([], maxlen=capacity)
        self.maxPrio = maxPrio
        self.alphaPrio = alphaPrio
        self.beta = beta_start
        self.betaStart = beta_start
        self.beta_frames = beta_frames
        self.frame_count = 0

    def push(self, *args):
      self.buffer.append(Transition(*args))
      if self.usePer :
        self.priority.append(self.maxPrio)

    def sample(self, batch_size):
      if self.usePer :

        self.frame_count += 1
        self.beta = min(1.0, self.betaStart + self.frame_count * (1.0-self.betaStart)/self.beta_frames)

        weightPrio = np.pow(np.array(self.priority),self.alphaPrio)
        weightPrio = weightPrio / np.sum(weightPrio)

        index = random.choices(range(len(self.buffer)), k=batch_size, weights=weightPrio)
        transitions = [self.buffer[i] for i in index]
        betaWeight = weightPrio[index]

        betaWeight = np.pow(1/(betaWeight * len(self.buffer)), self.beta)
        betaWeight = betaWeight / np.max(betaWeight)

        return transitions, index, betaWeight
      
      else :
        return random.sample(self.buffer, batch_size), None, None


    def updatePrio(self, deltas, indexs) :
      
      for i in range(len(indexs)) :
        self.priority[indexs[i]] = float(deltas[i, 0]) + 1e-5 


    def __len__(self):
      return len(self.buffer)
    

class SmallMemory(object):

    def __init__(self, nStep, gamma):
      self.gamma = gamma
      self.buffer = deque([], maxlen=nStep)

    def push(self, *args):
      
      if len(self.buffer) < self.buffer.maxlen :
        self.buffer.append(Transition(*args))
      else : 
        state = self.buffer[0].state
        action = self.buffer[0].action
        next_state = self.buffer[-1].next_state
        sum_reward = 0
        for i in range(self.buffer.maxlen) :
          sum_reward += self.buffer[i].reward*(self.gamma**i)

        gameIsOver = self.buffer[-1].gameIsOver
        prevTransition = Transition(state, action, next_state, sum_reward, gameIsOver)
        self.buffer.append(Transition(*args))
        return prevTransition

    def flush(self) :
      transitionsList = []
      while len(self.buffer) > 0 :
        state = self.buffer[0].state
        action = self.buffer[0].action
        next_state = self.buffer[-1].next_state
        sum_reward = 0
        for i in range(len(self.buffer)) :
          sum_reward += self.buffer[i].reward*(self.gamma**i)

        gameIsOver = self.buffer[-1].gameIsOver
        prevTransition = Transition(state, action, next_state, sum_reward, gameIsOver)
        self.buffer.popleft()
        transitionsList.append(prevTransition)
      return transitionsList

    def __len__(self):
      return len(self.buffer)
