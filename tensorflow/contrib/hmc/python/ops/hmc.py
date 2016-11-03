# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''Module for constructing Hamiltonian MonteCarlo'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def kinetic_energy(velocity):
	return 0.5*tf.square(velocity)

class Hamiltonian_MonteCarlo():
	def __init__(self,
				step_count,
				step_size,
				log_posterior,
				energy_func = kinetic_energy,
				pos,
				vel = 0):
	''' initizialize variables:
	
	Args:
		step_count = number of iterations
		step_size = magnitude of jump
		log_posterior = function
		velocity_func = hamiltonian physics
		pos0 = initial position
		vel0 = initial velocity
	'''
	
		self.steps = steps
		self.step_size = step_size
		self.log_posterior = log_posterior
		self.velocity_func = velocity_func
		self.pos = pos0
		self.vel = tf.random_normal(pos.get_shape())

	def __call__(self, step_size, num_steps, log_posterior, energy_func, pos, vel):
		vel = vel - 0.5 * step_size * tf.gradients(log_posterior(pos), pos)[0] #adjust velocity 1/2 step
		pos = pos + vel * step_size #update position

		for i in range(num_steps): #calculate gradients
			if i != num_steps-1:
				gradient = tf.gradients(log_posterior(pos), pos)[0] # calculate gradient
				vel = vel - step_size * gradient #update velocity
				x = x + step_size * vel #update position
		vel = vel - 0.5 *step_size * tf.gradients(log_posterior(pos), pos)[0] #adjust velocity final 1/2 step
		vel = -vel #negate to make symmetric
		return log_posterior(pos), kinetic_energy(vel)

	






