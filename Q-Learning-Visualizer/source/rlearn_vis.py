#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import os
import sys
import time
import curses
from turtle import color
import numpy
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

## Map Class
class Map:
  def __init__(self, filename):
    map_data = open(filename).readlines()
    # Set Cell Properties
    self.size_y = len(map_data)
    self.size_x = max([len(x.rstrip()) for x in map_data])
    # Map Properties
    self.starting_point = None
    self.reward = []
    self.grid = []

    # Load Map
    for row in range(self.size_y):
      # Add Data Row
      self.grid.append([])
      self.reward.append([])
      # Iterate and Calculate Points
      for grid in range(self.size_x):
        self.grid[row].append(map_data[row][grid])
        # Check Type of Score/Rewards
        if map_data[row][grid] == 'S':
            self.starting_point = (grid, row)
  
  ## Getter
  def getCell(self, x, y):
    # Identify Out of Bound Requests
    if x < 0 or x >= self.size_x or y < 0 or y >= self.size_y :
      return '!'
    return self.grid[y][x]

  def getStartingPoint(self):
    return self.starting_point

## Q Learning Policy Class
class QLearningPolicy:
  def __init__(self, epsilon, alpha, gamma):
    self.epsilon = epsilon
    self.alpha   = alpha
    self.gamma   = gamma

    # Initialize 4 Possible Actions
    self.actions = range(4)
    # Initialize Q Table
    self.q_table = {}

  # Making Decisions
  def MakeDecision(self, state):
    # If within probability of taking a random action 
    if random.random() < self.epsilon:
      action = random.choice(self.actions)
    else:
      all_actions = self.StateAllActions(state)
      # Get Highest Probability
      maximal_action = max(all_actions)
      # Check for Plausible Same Probability
      choices = all_actions.count(maximal_action)
      # If More than 1 Choice
      if choices > 1:
        best_choice = [i for i in range(4) if all_actions[i] == maximal_action]
        # Pick Random Best
        i = random.choice(best_choice)
      else:
        i = all_actions.index(maximal_action)

      action = self.actions[i]

    # Return Corresponding Action
    return action

  def StateAllActions(self, state):
    return [self.q_table.get((state, action), 0.0) for action in self.actions]
  
  # Q-Learning
  def LearnQValue(self, state, action, reward, value):
    old_value = self.q_table.get((state, action))
    # Check if There's Existing Values
    if old_value is None:
      self.q_table[(state, action)] = reward
    else:
      self.q_table[(state, action)] = old_value + self.alpha * (value - old_value)

  # General Agent Learning
  def Learn(self, state1, action1, reward, state2):
    # Obtain Best Course of Action
    maximal_q = max([self.q_table.get((state2, action), 0.0) for action in self.actions])
    self.LearnQValue(state1, action1, reward, reward + self.gamma * maximal_q)

## SARSA Policy Class
class SARSAPolicy:
  def __init__(self, epsilon, alpha, gamma):
    self.epsilon = epsilon
    self.alpha   = alpha
    self.gamma   = gamma

    # Initialize 4 Possible Actions
    self.actions = range(4)
    # Initialize Q Table
    self.q_table = {}

  # Making Decisions
  def MakeDecision(self, state):
    # If within probability of taking a random action 
    if random.random() < self.epsilon:
      action = random.choice(self.actions)
    else:
      all_actions = self.StateAllActions(state)
      # Get Highest Probability
      maximal_action = max(all_actions)
      # Check for Plausible Same Probability
      choices = all_actions.count(maximal_action)
      # If More than 1 Choice
      if choices > 1:
        best_choice = [i for i in range(4) if all_actions[i] == maximal_action]
        # Pick Random Best
        i = random.choice(best_choice)
      else:
        i = all_actions.index(maximal_action)

      action = self.actions[i]

    # Return Corresponding Action
    return action

  def StateAllActions(self, state):
    return [self.q_table.get((state, action), 0.0) for action in self.actions]
  
  # Q-Learning
  def LearnQValue(self, state, action, reward, value):
    old_value = self.q_table.get((state, action))
    # Check if There's Existing Values
    if old_value is None:
      self.q_table[(state, action)] = reward
    else:
      self.q_table[(state, action)] = old_value + self.alpha * (value - old_value)

  # General Agent Learning
  def Learn(self, state1, action1, reward, state2, action2):
    # Obtain Best Course of Action
    next_q = self.q_table.get((state2, action2), 0.0)
    self.LearnQValue(state1, action1, reward, reward + self.gamma * next_q)

  
## Policy Execution Agent
class Agent:
  def __init__(self, parameters, policy_type, map_data):
    # Check Policy Type and Initialize Accordingly
    if policy_type == 'qlearn':
      self.policy = QLearningPolicy(epsilon=parameters[0], alpha=parameters[1], gamma=parameters[2])
      self.policy_type = 'Q'
    else:
      self.policy = SARSAPolicy(epsilon=parameters[0], alpha=parameters[1], gamma=parameters[2])
      self.policy_type = 'S'

    self.prev_action = None
    self.score = 0
    self.map_data = map_data
    self.state = self.map_data.getStartingPoint()
    self.movement_policy = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    # Agent Statistics
    self.dead_count  = 1
    self.goal_count  = 1
  
  def update(self):
    curr_reward = self.__calculateReward()
    curr_action = self.policy.MakeDecision(self.state)
    # Check if It's Not First Action
    if self.prev_action is not None:
      if self.policy_type == 'Q':
        self.policy.Learn(self.prev_state, self.prev_action, curr_reward, self.state)
      else: # SARSA Learning
        self.policy.Learn(self.prev_state, self.prev_action, curr_reward, self.state, curr_action)
    # Set History
    self.prev_state = self.state
    self.prev_action = curr_action

    # Reduce Curiousity Over Time : Proposed 'decreasing-ε' using “time” in order
    # to reduce the exploration probability for better results as cited in :
    #
    # Caelen, O., Bontempi, G.: Improving the exploration strategy in bandit algorithms.
    # In: Learning and Intelligent Optimization. Number 5313 in LNCS.
    # Springer (2008) 56–68
    self.policy.epsilon -= 0.000001   ## Comment to Remove Policy

    # Make Movement
    curr_cell = self.__getCell()
    if curr_cell == 'X' or curr_cell == 'G':
      self.state = self.map_data.getStartingPoint()
      self.prev_action = None
      # Update Agent Statistics
      if curr_cell == 'X':
        self.dead_count += 1
      else:
        self.goal_count += 1
    else:
      self.__moveAgent(curr_action)

  # Bulk Update for Pretraining or Skipping
  def fastforward(self, iterations):
    for i in range(iterations):
      self.update()
  
  def __getCell(self):
    return self.map_data.getCell(self.state[0], self.state[1])

  def __moveAgent(self, action):
    # Set New Self State
    movement = self.movement_policy[action]
    target_state = (self.state[0] + movement[0], self.state[1] + movement[1])
    # Calculate Next Point
    target_cell  = self.map_data.getCell(target_state[0], target_state[1])
    # Check if Hit Wall
    if target_cell == '.' or target_cell == '!':
      return False

    self.state = target_state
    return True

  # Calculate Reward for Making Certain Move
  def __calculateReward(self):
    curr_cell = self.map_data.getCell(self.state[0], self.state[1])
    # Check Against Rewards Table
    if curr_cell == 'X':    # Cliff Cell
      return -7
    elif curr_cell == 'G':
      self.score += 1       # Improve One's Score
      return 1              # Return 0 for No Reward
    else:
      return -1             # Normal Reward

  # Agent Monitoring Function
  def monitor(self, screen):
    # Print Time Elapsed
    screen.addstr(0, 0, PrintTime())
    # Render Simulation Screen
    for row in range(1, self.map_data.size_y + 1):
      for col in range(0, self.map_data.size_x):
        screen.addstr(row, col, str(self.map_data.grid[row - 1][col]))
    # Add Agent Position on Map
    screen.addstr(self.state[1] + 1, self.state[0], str('@'))
    # Render Statistics
    q_arr = self.policy.StateAllActions(self.state)
    screen.addstr(self.map_data.size_y + 2, 0, 
                  str("Top: {:.3f}\tDown: {:.3f}\tLeft: {:.3f}\tRight: {:.3f}".format(q_arr[1], q_arr[0], q_arr[2], q_arr[3])))
    screen.addstr(self.map_data.size_y + 4, 0, str("Lifetime: " + str(self.dead_count + self.goal_count) + "\t\tSuccess Ratio: " + str((float)(self.goal_count)/(float)(self.dead_count))))
    screen.refresh()
  
    # Check for Escape Key
    capture_key = screen.getch()
    if capture_key == 27:
      return 0, False
    # Parse Capture Key to Main Loop
    elif capture_key == 32:
      return capture_key, True
    return 0, True

  # Agent Q Function
  def printQ(self, screen):
    rowmult=7
    # Render Simulation Screen
    for row in range(0, self.map_data.size_y):
      for col in range(0, self.map_data.size_x):
        curr_cell = self.map_data.getCell(col,row)
        Qval = self.policy.StateAllActions((col,row))
        if curr_cell in ['X','G','S']:
          screen.addstr(row*3+1 , col*3*rowmult+1*rowmult, str("  |"+curr_cell+"| "))
        if curr_cell != '.':
          screen.addstr(row*3+2, col*3*rowmult+1*rowmult, str("|{:.2f}|".format(Qval[0])))
          screen.addstr(row*3, col*3*rowmult+1*rowmult, str("|{:.2f}|".format(Qval[1])))
          screen.addstr(row*3+1, col*3*rowmult, str("|{:.2f}|".format(Qval[2])))
          screen.addstr(row*3+1, col*3*rowmult+2*rowmult, str("|{:.2f}|".format(Qval[3])))

          #screen.addstr(row*3+2, col*9+3, str("D1111"))
          #screen.addstr(row*3, col*9+3, str("U2222"))
          #screen.addstr(row*3+1, col*9, str("L33333"))
          #screen.addstr(row*3+1, col*9+6, str("R4444"))

    # Render Statistics
    q_arr = self.policy.StateAllActions(self.state)
    screen.addstr(self.map_data.size_y*3 + 8, 0, str("Top: {:.3f}\tDown: {:.3f}\tLeft: {:.3f}\tRight: {:.3f}".format(q_arr[1], q_arr[0], q_arr[2], q_arr[3])))
    screen.addstr(self.map_data.size_y*3 + 10, 0, str("Lifetime: " + str(self.dead_count + self.goal_count) + "\t\tSuccess Ratio: " + str((float)(self.goal_count)/(float)(self.dead_count))))
    screen.refresh()

    # Check for Escape Key
    capture_key = screen.getch()
    if capture_key == 27:
      return 0, False
    # Parse Capture Key to Main Loop
    elif capture_key == 32:
      return capture_key, True
    return 0, True

  def plotQ(self,figure, ax):
    norm = mpl.colors.Normalize(vmin=-10, vmax=1)
    cmap = cm.hot
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    #figure.clear()
    #L = np.zeros(self.)
    boxsize = 20
    boardsizex, boardsizey = (self.map_data.size_x*boxsize, self.map_data.size_y*boxsize)
    rectx, recty = np.meshgrid(np.linspace(0,boardsizex,self.map_data.size_x+1),np.linspace(0,boardsizey,self.map_data.size_y+1))

    for i in range(0,recty.shape[0]):
      ax.plot(rectx[i,:],recty[i,:], color='b')

    for i in range(0,rectx.shape[0]):
      ax.plot(rectx[:,i],recty[:,i], color='b')

    for i in range(-rectx.shape[0], rectx.shape[0]):
      ax.plot(np.diag(rectx,i),np.diag(recty,i), color='b')

    for i in range(-rectx.shape[0], rectx.shape[0]):
      ax.plot(np.diag(np.fliplr(rectx),i),np.diag(np.fliplr(recty),i), color='b')

    patchD = []
    patchL = []
    patchU = []
    patchR = []
    for j in range(0,recty.shape[0]-1):
      for i in range(0,rectx.shape[0]-1):
        x = [rectx[i,j], rectx[i,j]+boxsize, rectx[i,j] + boxsize/2]
        y = [recty[i,j], recty[i,j], recty[i,j]+boxsize/2]
        patchD.append(np.vstack((x,y)).T)

        x = [rectx[i,j],          rectx[i,j],         rectx[i,j] + boxsize/2]
        y = [recty[i,j],          recty[i,j]+boxsize, recty[i,j]+boxsize/2]
        patchL.append(np.vstack((x,y)).T)

        x = [rectx[i,j],          rectx[i,j]+boxsize, rectx[i,j] + boxsize/2]
        y = [recty[i,j]+boxsize,  recty[i,j]+boxsize, recty[i,j]+boxsize/2]
        patchU.append(np.vstack((x,y)).T)

        x = [rectx[i,j]+boxsize,  rectx[i,j]+boxsize, rectx[i,j] + boxsize/2]
        y = [recty[i,j],          recty[i,j]+boxsize, recty[i,j]+boxsize/2]
        patchR.append(np.vstack((x,y)).T)


    xoffset = [.45, .45, .05, .75]
    yoffset = [.85, .05, .45, .45]
    for row in range(0, self.map_data.size_y):
      for col in range(0, self.map_data.size_x):
        curr_cell = self.map_data.getCell(col,row)
        Qval = self.policy.StateAllActions((col,row))
        if curr_cell in ['X','G','S']:
          ax.text(rectx[row,col]+boxsize/2, recty[row,col]+boxsize/2, str("  |"+curr_cell+"| "))
        if curr_cell != '.':
          #screen.addstr(row*3+2, col*3*rowmult+1*rowmult, str("|{:.2f}|".format(Qval[0])))
          ax.text(rectx[row,col]+boxsize*xoffset[0], recty[row,col]+boxsize*yoffset[0], str("|{:.2f}|".format(Qval[0])))
          p = patchU[np.ravel_multi_index((col,row), (rectx.shape[0]-1,rectx.shape[1]-1))]
          ax.fill(p[:,0],p[:,1],color=m.to_rgba(Qval[0]))
          #screen.addstr(row*3, col*3*rowmult+1*rowmult, str("|{:.2f}|".format(Qval[1])))
          ax.text(rectx[row,col]+boxsize*xoffset[1], recty[row,col]+boxsize*yoffset[1], str("|{:.2f}|".format(Qval[1])))
          p = patchD[np.ravel_multi_index((col,row), (rectx.shape[0]-1,rectx.shape[1]-1))]
          ax.fill(p[:,0],p[:,1],color=m.to_rgba(Qval[1]))
          #screen.addstr(row*3+1, col*3*rowmult, str("|{:.2f}|".format(Qval[2])))
          ax.text(rectx[row,col]+boxsize*xoffset[2], recty[row,col]+boxsize*yoffset[2], str("|{:.2f}|".format(Qval[2])))
          p = patchL[np.ravel_multi_index((col,row), (rectx.shape[0]-1,rectx.shape[1]-1))]
          ax.fill(p[:,0],p[:,1],color=m.to_rgba(Qval[2]))
          #screen.addstr(row*3+1, col*3*rowmult+2*rowmult, str("|{:.2f}|".format(Qval[3])))
          ax.text(rectx[row,col]+boxsize*xoffset[3], recty[row,col]+boxsize*yoffset[3], str("|{:.2f}|".format(Qval[3])))
          p = patchR[np.ravel_multi_index((col,row), (rectx.shape[0]-1,rectx.shape[1]-1))]
          ax.fill(p[:,0],p[:,1],color=m.to_rgba(Qval[3]))

    figure.canvas.draw()
    #plt.draw()
    plt.pause(0.05)
    #plt.show()

    # Check for Escape Key
    capture_key = screen.getch()
    if capture_key == 27:
      return 0, False
    # Parse Capture Key to Main Loop
    elif capture_key == 32:
      return capture_key, True
    return 0, True


# Print Current Execution Time
def PrintTime():
  return "[" + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "]"

# Initialize Console Setup Teardown
def Teardown(screen):
    # reverse everything that you changed about the terminal
    curses.nocbreak()
    curses.echo()
    # restore the terminal to its original state
    curses.endwin()
    screen.clear()

# Main Function Entry Point
if __name__ == '__main__':
  # Load Cliff Map ( 'cliff.txt' as default )
  import os
  #print("directory:")
  #print(os.getcwd())
  #print(sys.argv[0])
  #print(sys.argv[1])
  cliff = Map("C:\\Users\\vaessenmj\\OneDrive - ZuydHogeschool\\Documents\\Python Scripts\\Q-Learning-Visualizer\\source\\cliff_simple.txt")
  #cliff = Map(sys.argv[1])
  # Create Training Agent
  my_agent = Agent([0.4, 0.1, 0.9], 'qlearn', cliff)

  figure, ax = plt.subplots()
  # Initialize Curses Screen 
  screen     = curses.initscr()
  #screen     = curses.newwin(50,150)
  
  monitoring = True
  sleep_time = 0.1
  screen.nodelay(True)

  # Perform Pretraining ( Set to Zero if Needed )
  my_agent.fastforward(0)

  # Perform Training
  while monitoring:
    
    my_agent.update()
    capture_key, monitoring = my_agent.plotQ(figure, ax)

    ## Press [SPACE] to Fast Forward Training by 50000 Updates
    if capture_key == 32:
      my_agent.fastforward(50000)
    elif capture_key == 97:
      break
      sleep_time *= 2
    elif capture_key == 100:
      sleep_time /= 2

    ## Visual Delay for Observations ( Lower is Faster )
    time.sleep(sleep_time)

  print(my_agent.state)
  # Clear Screen
  Teardown(screen)
    