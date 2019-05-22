import logging

import gym
from gym import spaces

CODE_DIREC_MAP = {0: 'for', 1: 'back', 2: 'left', 3: 'right'}
DECREASE_REWARD = 1
INCREASE_REWARD = -1
NOCHANGE_REWARD = 0

def todirec(code):
	return CODE_DIREC_MAP[code]
	
def tocode(direction):
	if direction == 'for':
		return 0
	if direction == 'back':
		return 1
	if direction == 'left':
		return 2
	return 3

def get_car_loc(image):
	# image processing and getting car location code or function
	# get y and x
	return y, x
	
def get_state() :
	# get current state(image)
	return state
	
def get_park_loc(image):
	# image processing and getting parking lot location code of function
	# get y and x
	return y, x
	
def after_action_state(state, action):  # state is image
	# capture the image(next state) after action at step
	# return next state
	return nstate
	
def check_game_status(timer):
	# check whether timeout or not
	return -1, -1  # still playing
	return time, reward  # timeout or distance = 0

class ParkingEnv(gym.Env):
	def __init__(self, alpha=0.02):
		self.alpha = alpha
		self.timer = 
		self.car_loc = 
		self.distance = 
		self.seed()
		self.reset()
		
	def reset(self):
		self.timer = 
		self.car_loc = 
		self.distance = 
		self.done = False
		return self._get_obs()
		
	def step(self, action) :
		direc = action
		if self.done:
			return self._get_obs(), reward, True, None
			
		reward = NOCHANGE_REWARD
		# do action for car
		status = check_game_status(self.board)[1]
		logging.debug("check_game_status status {}".format(status))
		
		# get act_reward
		after_action_distance = # distance after doing action
		act_reward = NOCHANGE_REWARD
		if self.distance > after_action_distance:
			act_reward = DECREASE_REWARD
		elif self.distance < after_action_distance:
			act_reward = INCREASE_REWARD
		
		self.distance = after_action_distance
		
		if status >= 0:
			self.done = True
			reward = status + act_reward
		
		return self._get_obs(), reward, self.done, None
		
	def _get_obs(self):
		return nstate
		
	def render(self, mode='car', close=False):
		if close:
			return
		if mode == 'car':
			# print screen
			
	def show_episode(self, car, episode):
		self._show_episode(print if car else logging.warning, episode)
		
	def _show_episode(self, showfn, episode):
		showfn("==== Episode {} ====".format(episode)):
		
	def show_result(self, car, reward):
		self._show_result(print if car else logging.info, reward)
		
	def _show_result(self, showfn, reward):
		status = check_game_status(timer)
		assert status[0] >= 0
		showfn("==== Time: {}, Reward: {} ====".format(status[0], status[1]))
		showfn('')
		
	def available_action(self):
		return # return 0, 1, 2, 3 except over the screen