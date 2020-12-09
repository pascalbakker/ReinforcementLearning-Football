from kaggle_environments.envs.football.helpers import *
from math import sqrt
from Actor import Critic, Actor

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
