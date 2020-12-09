import numpy as np
import os
import torch

torch.manual_seed(0)
np.random.seed(0)
import torch.nn.functional as F
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=8, kernel_size=8, stride=4, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(1280+4, 256, bias=True)
        self.fc2 = nn.Linear(256, 19)
        self.relu = nn.ReLU()
        self.b_1 = nn.BatchNorm2d(4)
        self.b_2 = nn.BatchNorm2d(8)
        self.b_3 = nn.BatchNorm2d(16)
        self.b_4 = nn.BatchNorm1d(1280+4)
        self.b_5 = nn.BatchNorm1d(256)

    def forward(self, x, scalar):
        x = torch.tensor(x).float()  # normalize
        x = x.permute(0, 3, 1, 2).contiguous()  # 1 x channels x height x width
        x = self.b_1(x)
        x = self.relu(self.conv1(x))
        x = self.b_2(x)
        x = self.relu(self.conv2(x))
        x = self.b_3(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.b_4(torch.cat([x, scalar], 1))
        x = self.relu(self.fc1(x))
        x = self.b_5(x)
        x = self.fc2(x)
        return F.softmax(x, dim = -1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=8, kernel_size=8, stride=4, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(1280+4, 256, bias=True)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.b_1 = nn.BatchNorm2d(4)
        self.b_2 = nn.BatchNorm2d(8)
        self.b_3 = nn.BatchNorm2d(16)
        self.b_4 = nn.BatchNorm1d(1280+4)
        self.b_5 = nn.BatchNorm1d(256)

    def forward(self, x, scalar):
        x = torch.tensor(x).float()  # normalize
        x = x.permute(0, 3, 1, 2).contiguous()  # 1 x channels x height x width
        x = self.b_1(x)
        x = self.relu(self.conv1(x))
        x = self.b_2(x)
        x = self.relu(self.conv2(x))
        x = self.b_3(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.b_4(torch.cat([x, scalar], 1))
        x = self.relu(self.fc1(x))
        x = self.b_5(x)
        x = self.fc2(x)
        return x
from kaggle_environments.envs.football.helpers import *
from math import sqrt

directions = [
[Action.TopLeft, Action.Top, Action.TopRight],
[Action.Left, Action.Idle, Action.Right],
[Action.BottomLeft, Action.Bottom, Action.BottomRight]]

dirsign = lambda x: 1 if abs(x) < 0.01 else (0 if x < 0 else 2)

enemyGoal = [1, 0]
perfectRange = [[0.61, 1], [-0.2, 0.2]]

def inside(pos, area):
    return area[0][0] <= pos[0] <= area[0][1] and area[1][0] <= pos[1] <= area[1][1]

def get_distance(pos1,pos2):
    return ((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)**0.5

def player_direction(obs):
    controlled_player_pos = obs['left_team'][obs['active']]
    controlled_player_dir = obs['left_team_direction'][obs['active']]
    x = controlled_player_pos[0]
    y = controlled_player_pos[1]
    dx = controlled_player_dir[0]
    dy = controlled_player_dir[1]

    if x <= dx:
        return 0
    if x > dx:
        return 1

def run_pass(left_team,right_team,x,y):
    ###Are there defenders dead ahead?
    defenders=0
    for i in range(len(right_team)):
        if right_team[i][0] > x and y +.01 >= right_team[i][1] and right_team[i][1]>= y - .01:
            if abs(right_team[i][0] - x) <.01:
                defenders=defenders+1
    if defenders == 0:
        return Action.Right

    teammateL=0
    teammateR=0
    for i in range(len(left_team)):
        #is there a teamate close to left
        if left_team[i][0] >= x:
            if left_team[i][1] < y:
                if abs(left_team[i][1] - x) <.05:
                    teammateL=teammateL+1

        #is there a teamate to right
        if left_team[i][0] >= x:
            if left_team[i][1] > y:
                if abs(left_team[i][1] - x) <.05:
                    teammateR=teammateR+1
    #pass only close to goal
    if x >.75:
        if teammateL > 0 or teammateR > 0:
            return Action.ShortPass

    if defenders > 0 and y>=0:
        return Action.TopRight

    if defenders > 0 and y<0:
        return Action.BottomRight


def agent(obs):
    controlled_player_pos = obs['left_team'][obs['active']]

    # special plays
    if obs["game_mode"] == GameMode.Penalty:
        return Action.Shot
    if obs["game_mode"] == GameMode.Corner:
        if controlled_player_pos[0] > 0:
            return Action.Shot
    if obs["game_mode"] == GameMode.FreeKick:
        return Action.Shot

    # Make sure player is running.
    if  0 < controlled_player_pos[0] < 0.6 and Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint
    elif 0.6 < controlled_player_pos[0] and Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint

    # Does the player we control have the ball?
    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:

        goalkeeper = 0
        #if in the zone near goal shoot
        if inside(controlled_player_pos, perfectRange) and controlled_player_pos[0] < obs['ball'][0]:
            return Action.Shot
        #if the goalie is coming out on player near goal shoot
        elif abs(obs['right_team'][goalkeeper][0] - 1) > 0.2 and controlled_player_pos[0] > 0.4 and abs(controlled_player_pos[1]) < 0.2:
            return Action.Shot
        # if close to goal and too wide for shot pass the ball
        if controlled_player_pos[0] >.75 and controlled_player_pos[1] >.20 or controlled_player_pos[0] >.75 and controlled_player_pos[1] <-.20 :
            return Action.ShortPass
        # if near our goal and moving away long pass to get out of our zone
        if player_direction(obs)==1 and controlled_player_pos[0]<-.3:
            return Action.LongPass
        # which way should we run or pass
        else:
            return run_pass(obs['left_team'],obs['right_team'],controlled_player_pos[0],controlled_player_pos[1])
    else:
        #vector where ball is going
        ball_targetx=obs['ball'][0]+obs['ball_direction'][0]
        ball_targety=obs['ball'][1]+obs['ball_direction'][1]

        #euclidian distance to the ball so we head off movement until very close
        e_dist=get_distance(obs['left_team'][obs['active']],obs['ball'])

        #if not close to ball move to where it is going
        if e_dist >.005:
            # Run where ball will be
            xdir = dirsign(ball_targetx - controlled_player_pos[0])
            ydir = dirsign(ball_targety - controlled_player_pos[1])
            return directions[ydir][xdir]
        #if close to ball go to ball
        else:
            # Run towards the ball.
            xdir = dirsign(obs['ball'][0] - controlled_player_pos[0])
            ydir = dirsign(obs['ball'][1] - controlled_player_pos[1])
            return directions[ydir][xdir]


w_step = 2/96.0
h_step = 0.84/72

def get_coordinates(arr):
    x, y = arr
    x_i = 0
    y_i = 0
    for i in range(1, 96):
        if x <-1 or x>1:
            if x<-1:
                x_i = 0
            else:
                x_i = 95
        else:
            if -1+ (i-1)*w_step <= x <= -1 + i*w_step:
                x_i = i
                break

    for i in range(1, 72):
        if y <-0.42 or y>0.42:
            if y<-0.42:
                y_i = 0
            else:
                y_i = 71
        else:
            if -0.42+ (i-1)*h_step <= y <= -0.42 + i*h_step:
                y_i = i
                break
    return [y_i, x_i]


def get_team_coordinates(team_arr):
    answ = []
    for j in range(len(team_arr)):
        answ.append(get_coordinates(team_arr[j]))
    return answ


import math
import numpy as np

def angle(src, tgt):
    dx = tgt[0] - src[0]
    dy = tgt[1] - src[1]
    theta = round(math.atan2(dx, -dy) * 180 / math.pi, 2)
    while theta < 0:
        theta += 360
    return theta

def direction(src, tgt):
    actions = [3, 4, 5,
               6,  7,
               8, 1, 2]
    theta = angle(src, tgt)
    index = int(((theta+45/2)%360)/45)
    return actions[index]


def create_obs(obs):
    ball_coord = get_coordinates(obs['ball'][:-1])
    left_team_coord = get_team_coordinates(obs['left_team'])
    right_team_coord = get_team_coordinates(obs['right_team'])
    player_coord =  get_coordinates(obs['left_team'][obs['active']])


    obs_1 = np.zeros(shape = (1, 72, 96, 4))

    obs_1[0, ball_coord[0], ball_coord[1], 0] = 1

    obs_1[0, player_coord[0], player_coord[1], 0] = 1

    for i, l in enumerate(left_team_coord):

        obs_1[0, l[0], l[1], 2] = 1

    for i, r in enumerate(right_team_coord):
        obs_1[0, r[0], r[1], 3] = 1

    ball_next_coord = get_coordinates([obs['ball'][0] + obs['ball_direction'][0], obs['ball'][1] + obs['ball_direction'][1]])

    left_team_next_coord = []
    for i in range(len(obs['left_team'])):
        left_team_next_coord.append([obs['left_team'][i][0] + obs['left_team_direction'][i][0], obs['left_team'][i][1] + obs['left_team_direction'][i][1]])

    right_team_next_coord = []
    for i in range(len(obs['right_team'])):
        right_team_next_coord.append([obs['right_team'][i][0] + obs['right_team_direction'][i][0], obs['right_team'][i][1] + obs['right_team_direction'][i][1]])


    scalar = np.zeros(shape = (1, 4))
    scalar[0,0] = obs['ball_owned_team']
    scalar[0,1] = obs['game_mode']
    scalar[0,2] = direction(obs['ball'][:-1], obs['ball_direction'][:-1])
    scalar[0,3] = direction(obs['left_team'][obs['active']], obs['left_team_direction'][obs['active']])

    return obs_1, scalar

import torch
import torch.nn as nn
import numpy as np
import gfootball.env as football_env
from gfootball.env import observation_preprocessing





env = football_env.create_environment(
   env_name="11_vs_11_kaggle",
   representation='raw',
   stacked=False,
   logdir='.',
   write_goal_dumps=False,
   write_full_episode_dumps=False,
   render=False,
   number_of_left_players_agent_controls=1,
   dump_frequency=0)

obs = env.reset()

created_obs = create_obs(obs[0])
actor = Actor()
critic = Critic()

env  = football_env.create_environment(
   env_name="11_vs_11_kaggle",
   representation='raw',
   stacked=False,
   logdir='.',
   write_goal_dumps=False,
   write_full_episode_dumps=False,
   render=False,
   number_of_left_players_agent_controls=1,
   number_of_right_players_agent_controls=1,
   dump_frequency=0)

obs = env.reset()



adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99


step_done = 0
rewards_for_plot = []
for steps_done in range(64):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    games_play = 0
    wins = 0
    loses = 0
    obs = env.reset()
    values = []
    log_probs = []
    done = False
    while not done:

        converted_obs = create_obs(obs[0])
        actor.eval()
        prob = actor(torch.as_tensor(converted_obs[0], dtype = torch.float32), torch.as_tensor(converted_obs[1], dtype = torch.float32))
        actor.train()
        dist = torch.distributions.Categorical(probs = prob)
        act = dist.sample()


        new_obs, reward, done, _ = env.step([act.detach().data.numpy()[0], (agent(obs[1])).value])
        if reward[0]==-1:
            loses+=1
            done = True
        if reward[0] == 1:
            wins+=1
            done = True
        if reward[0]==0 and done:
            reward[0] = 0.25

        last_q_val = 0
        if done:
            converted_next_obs = create_obs(new_obs[0])
            critic.eval()
            last_q_val = critic(torch.as_tensor(converted_next_obs[0], dtype = torch.float32), torch.as_tensor(converted_next_obs[1], dtype = torch.float32))
            last_q_val = last_q_val.detach().data.numpy()
            critic.train()

        states.append(obs[0])
        action_arr = np.zeros(19)
        action_arr[act] = 1
        actions.append(action_arr)
        rewards.append(reward[0])
        next_states.append(new_obs[0])
        dones.append(1 - int(done))

        obs = new_obs
        if done:
            obs = env.reset()
            break

    rewards = np.array(rewards)
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    dones = np.array(dones)

    print('epoch '+ str(steps_done)+ '\t' +'Mean Reward' + str(np.mean(rewards)) + '\t' + 'Games Count ' + str(games_play) + '\t' + 'Total Games: ' + str(wins) + '\t'+ 'Total Loses ' + str(loses))
    rewards_for_plot.append(np.mean(rewards))
    #train
    q_vals = np.zeros((len(rewards), 1))
    for i in range(len(rewards)-1, 0, -1):
        last_q_val = rewards[i] + dones[i]*gamma*last_q_val
        q_vals[i] = last_q_val

    action_tensor = torch.as_tensor(actions, dtype=torch.float32)

    obs_playgraund_tensor = torch.as_tensor(np.array([create_obs(states[i])[0][0] for i in range(len(rewards))]), dtype=torch.float32)

    obs_scalar_tensor = torch.as_tensor(np.array([create_obs(states[i])[1][0] for i in range(len(rewards))]), dtype=torch.float32)

    val = critic(obs_playgraund_tensor, obs_scalar_tensor)

    probs = actor(obs_playgraund_tensor, obs_scalar_tensor)

    advantage = torch.Tensor(q_vals) - val

    critic_loss = advantage.pow(2).mean()
    adam_critic.zero_grad()
    critic_loss.backward()
    adam_critic.step()


    actor_loss = (-torch.log(probs)*advantage.detach()).mean()
    adam_actor.zero_grad()
    actor_loss.backward(retain_graph=True)
    adam_actor.step()



#         soft_update(actor, target_actor, 0.8)
#         soft_update(critic, target_critic, 0.8)

    if steps_done!=0 and steps_done%50 == 0:
        torch.save(actor.state_dict(), 'actor.pth')

        torch.save(critic.state_dict(), 'critic.pth')

torch.save(actor.state_dict(), 'actor.pth')

torch.save(critic.state_dict(), 'critic.pth')
