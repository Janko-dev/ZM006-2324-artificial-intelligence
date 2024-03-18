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
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.animation as anim


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
  def __init__(self, parameters, policy_type, map_data, goal, cliff, penalty, boxsize):
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

    self.goal = goal
    self.cliff = cliff
    self.penalty = penalty

    #for plotting
    self.boxsize = boxsize
    boardsizex, boardsizey = (self.map_data.size_x*boxsize, self.map_data.size_y*boxsize)
    self.rectx, self.recty = np.meshgrid(np.linspace(0,boardsizex,self.map_data.size_x+1),np.linspace(0,boardsizey,self.map_data.size_y+1),indexing='ij')

    self.playerplot = plt.Circle((self.rectx[0,0]+boxsize/2, self.recty[0,0]+boxsize/2),3,color='k',zorder=1000)
  
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
      return self.cliff
    elif curr_cell == 'G':
      self.score += 1       # Improve One's Score
      return self.goal              # Return 0 for No Reward
    else:
      return self.penalty             # Normal Reward

  #make the plot once. Most elements only have to be drawn at the start. Others get their properties updated during the run in the plotQ function
  def makeplotQ(self, figure, ax):
    #figure.clear()
    #L = np.zeros(self.)
    rectx = self.rectx
    recty = self.recty
    boxsize = self.boxsize

    for i in range(1,recty.shape[0]-1):
      ax.plot(rectx[i,1:-1],recty[i,1:-1], color='k')

    for i in range(1,rectx.shape[1]-1):
      ax.plot(rectx[1:-1,i],recty[1:-1,i], color='k')

    for i in range(-rectx.shape[0], rectx.shape[0]):
      ax.plot(np.diag(rectx,i)[1:-1],np.diag(recty,i)[1:-1], color='k')

    for i in range(-rectx.shape[0], rectx.shape[0]):
      ax.plot(np.diag(np.fliplr(rectx),i)[1:-1],np.diag(np.fliplr(recty),i)[1:-1], color='k')

    patchD = []
    patchL = []
    patchU = []
    patchR = []
    for i in range(0,rectx.shape[0]-1):
      for j in range(0,rectx.shape[1]-1):
        x = [rectx[i,j],          rectx[i,j]+boxsize, rectx[i,j] + boxsize/2]
        y = [recty[i,j],          recty[i,j],         recty[i,j] + boxsize/2]
        patchD.append(np.vstack((x,y)).T)

        x = [rectx[i,j],          rectx[i,j],         rectx[i,j] + boxsize/2]
        y = [recty[i,j],          recty[i,j]+boxsize, recty[i,j] + boxsize/2]
        patchL.append(np.vstack((x,y)).T)

        x = [rectx[i,j],          rectx[i,j]+boxsize, rectx[i,j] + boxsize/2]
        y = [recty[i,j]+boxsize,  recty[i,j]+boxsize, recty[i,j] + boxsize/2]
        patchU.append(np.vstack((x,y)).T)

        x = [rectx[i,j]+boxsize,  rectx[i,j]+boxsize, rectx[i,j] + boxsize/2]
        y = [recty[i,j],          recty[i,j]+boxsize, recty[i,j] + boxsize/2]
        patchR.append(np.vstack((x,y)).T)

    patchlst = []
    xoffset = [.45, .45, .05, .75]
    yoffset = [.85, .05, .45, .45]
    for row in range(0, self.map_data.size_x):
      for col in range(0, self.map_data.size_y):
        curr_cell = self.map_data.getCell(row,col)
        Qval = self.policy.StateAllActions((row,col))
        if curr_cell in ['S']:
          ax.add_patch(self.playerplot)
        if curr_cell in ['G','X','S']:
          if curr_cell == 'G':
            ax.add_patch(plt.Circle((rectx[row,col]+boxsize/2, recty[row,col]+boxsize/2),2,color='g',zorder=1000))
          if curr_cell == 'X':
            ax.add_patch(plt.Circle((rectx[row,col]+boxsize/2, recty[row,col]+boxsize/2),2,color='r',zorder=1001))
          if curr_cell == 'S':
            ax.add_patch(plt.Circle((rectx[row,col]+boxsize/2, recty[row,col]+boxsize/2),2,color='b',zorder=1002))
          ax.text(rectx[row,col]+boxsize/2, recty[row,col]+boxsize/2, str("|"+curr_cell+"| "))
        if curr_cell not in ['.','X','G']:
          #screen.addstr(row*3+2, col*3*rowmult+1*rowmult, str("|{:.2f}|".format(Qval[0])))
          ax.text(rectx[row,col]+boxsize*xoffset[0], recty[row,col]+boxsize*yoffset[0], str("|{:.2f}|".format(Qval[0])))
          p = patchU[np.ravel_multi_index((row,col), (rectx.shape[0]-1,rectx.shape[1]-1))]
          patchlst.append(Polygon(p))
          #screen.addstr(row*3, col*3*rowmult+1*rowmult, str("|{:.2f}|".format(Qval[1])))
          ax.text(rectx[row,col]+boxsize*xoffset[1], recty[row,col]+boxsize*yoffset[1], str("|{:.2f}|".format(Qval[1])))
          p = patchD[np.ravel_multi_index((row,col), (rectx.shape[0]-1,rectx.shape[1]-1))]
          patchlst.append(Polygon(p))
          #screen.addstr(row*3+1, col*3*rowmult, str("|{:.2f}|".format(Qval[2])))
          ax.text(rectx[row,col]+boxsize*xoffset[2], recty[row,col]+boxsize*yoffset[2], str("|{:.2f}|".format(Qval[2])))
          p = patchL[np.ravel_multi_index((row,col), (rectx.shape[0]-1,rectx.shape[1]-1))]
          patchlst.append(Polygon(p))
          #screen.addstr(row*3+1, col*3*rowmult+2*rowmult, str("|{:.2f}|".format(Qval[3])))
          ax.text(rectx[row,col]+boxsize*xoffset[3], recty[row,col]+boxsize*yoffset[3], str("|{:.2f}|".format(Qval[3])))
          p = patchR[np.ravel_multi_index((row,col), (rectx.shape[0]-1,rectx.shape[1]-1))]
          patchlst.append(Polygon(p))

    collection = PatchCollection(patchlst)
    ax.add_collection(collection)

    return collection

  #funtion to dynamically update the plot
  def plotQ(self,figure, ax, collection):
    norm = mpl.colors.Normalize(vmin=-2, vmax=0)
    cmap = cm.winter
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    cnt=0
    for row in range(0, self.map_data.size_x):
      for col in range(0, self.map_data.size_y):
        curr_cell = self.map_data.getCell(row,col)
        Qval = self.policy.StateAllActions((row,col))
        if self.state == (row,col):
          self.playerplot.set_center((self.rectx[row,col]+boxsize/2, self.recty[row,col]+boxsize/2))
        if curr_cell in ['G','X','S']:
          ax.texts[cnt].set_text(str("|"+curr_cell+"|"))
          cnt+=1
        if curr_cell not in ['.','X','G']:
          ax.texts[cnt].set_text(str("|{:.2f}|".format(Qval[0])))
          cnt+=1
          ax.texts[cnt].set_text(str("|{:.2f}|".format(Qval[1])))
          cnt+=1
          ax.texts[cnt].set_text(str("|{:.2f}|".format(Qval[2])))
          cnt+=1
          ax.texts[cnt].set_text(str("|{:.2f}|".format(Qval[3])))
          cnt+=1

    colors=[]
    for row in range(0, self.map_data.size_x):
      for col in range(0, self.map_data.size_y):
        curr_cell = self.map_data.getCell(row,col)
        Qval = self.policy.StateAllActions((row,col))
        if curr_cell not in ['.','X','G']:
          for i in range(4):
            colors.append(m.to_rgba(Qval[i]))
    #collection.set_array(np.asarray(colors))
    collection.set_facecolors(colors)

    #figure.canvas.draw()
    #plt.draw()
    #plt.pause(0.001)
    #plt.show()

    # Check for Escape Key
    capture_key = screen.getch()
    if capture_key == 27:
      return capture_key, False
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
  cliff = Map("cliff2.txt")
  #cliff = Map(sys.argv[1])
  # Create Training Agent
  a_epsilon=0.01
  a_alpha=0.5
  a_gamma=0.9
  a_reward_goal = 1
  a_reward_cliff = -100
  a_reward_step = -.1
  parameters = [a_epsilon, a_alpha, a_gamma]

  boxsize=15 #parameter for plot
  
  my_agent = Agent(parameters, 'qlearn', cliff,a_reward_goal,a_reward_cliff,a_reward_step, boxsize)

  figure, ax = plt.subplots()
  mpl.rcParams.update({'font.size': 8})
  
  # Initialize Curses Screen 
  screen     = curses.initscr()
  #screen     = curses.newwin(50,150)
  
  monitoring = True
  sleep_time = 100
  screen.nodelay(True)

  # Perform Pretraining ( Set to Zero if Needed )
  my_agent.fastforward(10)
  collection = my_agent.makeplotQ(figure, ax)
  # Perform Training
  while monitoring:
    
    def update(i):
      my_agent.update()
      capture_key, monitoring = my_agent.plotQ(figure, ax, collection)
      ## Press [SPACE] to Fast Forward Training by 50000 Updates
      if capture_key == 32:
        my_agent.fastforward(50000)
      elif capture_key == 27:
        exit()
        sleep_time *= 2
      elif capture_key == 100: 
        sleep_time /= 2

    a = anim.FuncAnimation(figure, update, interval=sleep_time)
    plt.show()

    ## Visual Delay for Observations ( Lower is Faster )
    #time.sleep(sleep_time)

  print(my_agent.state)
  # Clear Screen
  Teardown(screen)
  exit()
    