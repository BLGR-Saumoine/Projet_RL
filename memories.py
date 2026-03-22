from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'gameIsOver'))

class ReplayMemory(object):

    def __init__(self, capacity):
      self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
      self.buffer.append(Transition(*args))

    def sample(self, batch_size):
      return random.sample(self.buffer, batch_size)

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
