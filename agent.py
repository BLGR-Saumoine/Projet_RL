import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from dqnModel import DQN

class Agent:
    def __init__(self, n_FrameStack, n_actions, lr, gamma, tau, device, epsDecay, epsEnd, epsStart, use_double_dqn=True):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions

        self.epsDecay = epsDecay
        self.epsEnd = epsEnd
        self.epsStart = epsStart
        self.numberStep = 0
        
        self.use_double_dqn = use_double_dqn 

        self.onlineNetwork = DQN(n_FrameStack, n_actions).to(self.device)
        self.targetNetwork = DQN(n_FrameStack, n_actions).to(self.device)
        self.targetNetwork.load_state_dict(self.onlineNetwork.state_dict())

        self.optimizer = optim.AdamW(self.onlineNetwork.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

    def calculateQlearning(self, batch_nextState, batch_done, batch_reward, nStep) :

        with torch.no_grad() :
            next_predictions = self.targetNetwork(batch_nextState)
            maxQvalue = torch.max(next_predictions, dim=1).values
            bellmanTarget = batch_reward + (self.gamma**nStep) * (maxQvalue*~batch_done) #Multiplied by the inversed boolean : if game is finished -> True = 1, by inverting i assign 0 as a value when the game is finished
            
        return bellmanTarget
    
    def calculateDoublelearning(self, batch_nextState, batch_done, batch_reward, nStep) :

        with torch.no_grad() :
            next_predictions = self.onlineNetwork(batch_nextState)
            otherPrediction = self.targetNetwork(batch_nextState)
        
            maxQvalue = torch.gather(otherPrediction, dim=1, index=torch.argmax(next_predictions, dim=1).unsqueeze(1)).squeeze(1)

            bellmanTarget = batch_reward + (self.gamma**nStep) * (maxQvalue*~batch_done) #Multiplied by the inversed boolean : if game is finished -> True = 1, by inverting i assign 0 as a value when the game is finished
        
        return bellmanTarget


    def select_action(self, state):
        epsilon = self.epsEnd + (self.epsStart - self.epsEnd) * math.exp(-self.numberStep/self.epsDecay)

        if torch.rand(1).item() < epsilon :
            action = torch.randint(0,self.n_actions, (1,)).item()

        else :
            with torch.no_grad():
                gpu_state = state.to(self.device, dtype=torch.float32) / 255.0
                action = torch.argmax(self.onlineNetwork(gpu_state)).item()
                #action = torch.argmax(self.onlineNetwork(state)).item()
        self.numberStep += 1
        return action


    def optimizeModel(self, memory, batch_size, nStep):

        if len(memory) < batch_size:
            return None, None
        state = []
        action = []
        nextState = []
        reward = []
        done = []

        selectedTuples, index, betaWeight = memory.sample(batch_size)

        for tuples in selectedTuples :
            state.append(tuples.state)
            action.append(tuples.action)
            nextState.append(tuples.next_state)
            reward.append(tuples.reward)
            done.append(tuples.gameIsOver)

        batch_state = torch.cat(state).to(self.device, dtype=torch.float32) / 255.0
        batch_action = torch.tensor(action, device=self.device)
        batch_nextState = torch.cat(nextState).to(self.device, dtype=torch.float32) / 255.0
        batch_reward = torch.tensor(reward, device=self.device)
        batch_done = torch.tensor(done, dtype=torch.bool, device=self.device)

        if self.use_double_dqn :
            bellmanTarget = self.calculateDoublelearning( batch_nextState, batch_done, batch_reward, nStep)
        else :
            bellmanTarget = self.calculateQlearning(batch_nextState, batch_done, batch_reward, nStep)

        actual_predictions = self.onlineNetwork(batch_state)

        policyPrediction = torch.gather(actual_predictions, 1, batch_action.unsqueeze(1))

        if memory.usePer :
            criterion = nn.SmoothL1Loss(reduction='none')
            errors = criterion(policyPrediction, bellmanTarget.unsqueeze(1))
      
            weights_tensor = torch.tensor(betaWeight, device=self.device, dtype=torch.float32).unsqueeze(1)
            loss = (errors * weights_tensor).mean()
            
            with torch.no_grad():
                new_deltas = torch.abs(policyPrediction - bellmanTarget.unsqueeze(1)).detach().cpu().numpy()
            memory.updatePrio(new_deltas, index)

        else : 
            criterion = nn.SmoothL1Loss()
            loss = criterion(policyPrediction, bellmanTarget.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.onlineNetwork.parameters(), 100)
        self.optimizer.step()

        return torch.abs(policyPrediction - bellmanTarget.unsqueeze(1)).detach().cpu().numpy(), index #Monstrueux mais j'ai pas trouvé mieux pour le moment


    def softUpdateNetwork(self):
        target_net_state_dict = self.targetNetwork.state_dict()
        policy_net_state_dict = self.onlineNetwork.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.targetNetwork.load_state_dict(target_net_state_dict)
