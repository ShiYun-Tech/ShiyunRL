import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque

def ensure_directory_exists(path):
	if not os.path.exists(path):
		os.makedirs(path)
		print(f"Directory created: {path}")
	else:
		print(f"Directory already exists: {path}")
		
def smooth(input, window_size=20):
	# 使用窗口大小为 window_size 的移动平均平滑奖励
	smoothed_input = np.convolve(input, np.ones(window_size)/window_size, mode='valid')
	return smoothed_input

class MemoryBuffer(object):
	def __init__(self, size):
		self.buffer = deque(maxlen=size)
		self.maxSize = size
		self.len = 0

	def sample(self, count):
		batch = []
		count = min(count, self.len)
		batch = random.sample(self.buffer, count)
		s_arr = np.array([arr[0] for arr in batch], dtype=np.float32)
		a_arr = np.array([arr[1] for arr in batch], dtype=np.int64)
		a_log_arr = np.array([arr[2] for arr in batch], dtype=np.float32)
		r_arr = np.array([arr[3] for arr in batch], dtype=np.float32)
		s1_arr = np.array([arr[4] for arr in batch], dtype=np.float32)
		return s_arr, a_arr,a_log_arr, r_arr, s1_arr


	def getAll(self):
		s_arr = np.array([arr[0] for arr in self.buffer], dtype=np.float32)
		a_arr = np.array([arr[1] for arr in self.buffer], dtype=np.int64)
		a_log_arr = np.array([arr[2] for arr in self.buffer], dtype=np.float32)
		r_arr = np.array([arr[3] for arr in self.buffer], dtype=np.float32)
		s1_arr = np.array([arr[4] for arr in self.buffer], dtype=np.float32)
		return s_arr, a_arr,a_log_arr, r_arr, s1_arr
	
	def len(self):
		return self.len

	def add(self, state, action, a_log_prob,reward, next_state):
		s_arr = np.array(state, dtype=np.float32)
		a_arr = np.array(action, dtype=np.int64)
		a_log_arr = np.array(a_log_prob, dtype=np.float32)
		r_arr = np.array(reward, dtype=np.float32)
		s1_arr = np.array(next_state, dtype=np.float32)
		transition = (s_arr, a_arr, a_log_arr, r_arr, s1_arr)
		self.len += 1
		if self.len > self.maxSize:
			self.len = self.maxSize
		self.buffer.append(transition)
	def clear(self): # Add this method to clear the memory buffer
		self.buffer.clear()  # Clear the deque
		self.len = 0		 # Reset the length of memory