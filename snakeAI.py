import pygame
import sys
import random
import time
import numpy as np
from copy import deepcopy

snake_pop = 200
frame_rate = 120
nn_drive = True
dimensions = [190, 190]

# ========== Classes ==========
class NeuralNet():
	def __init__(self, inputs, h_num, out_num):
		self.iNodes = inputs
		self.oNodes = out_num
		self.hNodes = h_num
		#
		#Synapses
		# Always random, no seed
		self.syn0 = (2*np.random.random((self.iNodes+1,self.hNodes)) - 1) *1
		self.syn1 = (2*np.random.random((self.hNodes+1,self.hNodes)) - 1) *1
		self.syn2 = (2*np.random.random((self.hNodes+1,self.oNodes)) - 1) *1
		#
	#
	def __sig(self, x):
		return 1/(1+np.exp(-x))
	#
	def __dsig(self, x):
		return x*(1-x)
	#
	def addBias(self, nparray):
		rows = len(nparray)
		bias = np.ones((rows, 1))
		new_array = np.concatenate((nparray, bias), axis=1)
		return new_array
	#
	def output(self, inputs):
		l0 = np.array([inputs])
		l1 = self.__sig(np.dot(self.addBias(l0), self.syn0))
		l2 = self.__sig(np.dot(self.addBias(l1), self.syn1))
		l3 = self.__sig(np.dot(self.addBias(l2), self.syn2))
		return l3
	#
	def decision(self, inputs):
		decision = self.output(inputs)
		directions = {0:"LEFT", 1:"RIGHT", 2:"STRAIGHT", 3:"DIE"}
		return directions[np.argmax(decision)]
	#
	def mutateSyn(self, matrix, m_value):
		for i in range(len(matrix)):
			for j in range(len(matrix[i])):
				if np.random.random() < m_value:
					m = np.random.random()
					#m = (m*2-1)/5
					#m = 1 + (np.random.random()-.5)*3 + (np.random.random() -.5)
					#
					coin = np.random.choice([-1,1])
					m = (m*2-1)
					#
					matrix[i,j] += m
					"""
					if matrix[i,j] > 1:
						matrix[i,j] = 1
					if matrix[i,j] < -1:
						matrix[i,j] = -1
					"""
		return matrix
	#
	def mutate(self, m_value):
		self.syn0 = self.mutateSyn(self.syn0, m_value)
		self.syn1 = self.mutateSyn(self.syn1, m_value)
		self.syn2 = self.mutateSyn(self.syn2, m_value)

class Snake():
	def __init__(self):
		self.id = str(int(np.random.random()*5000))
		self.setBrain(24,18,4)
		self.setColor(np.random.random(3)*255)
		self.reset()
	#
	def reset(self):
		self.position = [100,50]
		self.body = [[100,50],[90,50],[80,50]]
		self.direction = "RIGHT"
		self.changeDirectionTo = self.direction
		self.life_time = 0
		self.ttl = int(np.average(dimensions)/2)
		self.score = 0
		self.distance = 1
		self.spawnFood()
	#
	def spawnFood(self):
		dimx = (dimensions[0]+10)/10
		dimy = (dimensions[1]+10)/10
		self.foodPos = [random.randrange(1,dimx)*10, random.randrange(1,dimy)*10]
	#
	def ateApple(self):
		self.score += 1
		#self.ttl += int(np.average(dimensions)/4)
		self.ttl = int(np.average(dimensions)/2)
		self.spawnFood()
	#
	def setColor(self, color):
		self.color = color
	#
	def setBrain(self, i, h, o):
		self.brain = NeuralNet(i,h,o)
	#
	def changeDirTo(self, dir):
		if dir=="RIGHT" and not self.direction=="LEFT":
			self.direction = "RIGHT"
		if dir=="LEFT" and not self.direction=="RIGHT":
			self.direction = "LEFT"
		if dir=="UP" and not self.direction=="DOWN":
			self.direction = "UP"
		if dir=="DOWN" and not self.direction=="UP":
			self.direction = "DOWN"
	#
	def changeDirTo_(self, dir):
		if dir=="RIGHT":
			self.direction = "RIGHT"
		if dir=="LEFT":
			self.direction = "LEFT"
		if dir=="UP":
			self.direction = "UP"
		if dir=="DOWN":
			self.direction = "DOWN"
	#
	def nnCD(self, dir):
		if dir == "STRAIGHT":
			self.distance += 1
		if self.direction == "RIGHT":
			if dir == "RIGHT":
				self.direction = "DOWN"
			elif dir == "LEFT":
				self.direction = "UP"
			elif dir == "DIE":
				self.direction = "LEFT"
			elif dir == "STRIAGHT":
				self.direction = self.direction
		elif self.direction == "LEFT":
			if dir == "RIGHT":
				self.direction = "UP"
			elif dir == "LEFT":
				self.direction = "DOWN"
			elif dir == "DIE":
				self.direction = "RIGHT"
			elif dir == "STRIAGHT":
				self.direction = self.direction
		elif self.direction == "UP":
			if dir == "RIGHT":
				self.direction = "RIGHT"
			elif dir == "LEFT":
				self.direction = "LEFT"
			elif dir == "DIE":
				self.direction = "UP"
			elif dir == "STRIAGHT":
				self.direction = self.direction
		elif self.direction == "DOWN":
			if dir == "RIGHT":
				self.direction = "LEFT"
			elif dir == "LEFT":
				self.direction = "RIGHT"
			elif dir == "DIE":
				self.direction = "DOWN"
			elif dir == "STRIAGHT":
				self.direction = self.direction
	#
	def move(self):
		if self.direction == "RIGHT":
			self.position[0] += 10
		if self.direction == "LEFT":
			self.position[0] -= 10
		if self.direction == "UP":
			self.position[1] -= 10
		if self.direction == "DOWN":
			self.position[1] += 10
		self.body.insert(0, list(self.position))
		self.life_time += 1
		self.ttl -= 1
		if self.position == self.foodPos:
			self.ateApple()
		else:
			self.body.pop()
	#
	def checkCollision(self):
		if self.position[0] > dimensions[0] or self.position[0] < 0:
			return 1
		elif self.position[1] > dimensions[1] or self.position[1] < 0:
			return 1
		for bodyPart in self.body[1:]:
			if self.position == bodyPart:
				return 1
		if self.ttl <= 0:
			return 1
		return 0
	#
	def getHeadPos(self):
		return self.position
	#
	def getBody(self):
		return self.body
	#
	def lookInDirection(self, direction):
		in_view = [0,0,0]
		distance = 0
		direction = np.array(direction)
		food = False
		tail = False
		# move once
		view_space = direction + self.position
		distance += 1
		#if str(direction) == "[10  0]":
		#	print view_space
		# While viewspace is not touching a wall
		while (view_space[0] <= dimensions[0]\
			and view_space[0] >= 0\
			and view_space[1] <= dimensions[1]\
			and view_space[1] >= 0):
				if self.foodPos == view_space.tolist():
					in_view[0] = 1
				if view_space.tolist() in self.body:
					in_view[1] = 1./distance
					break
				view_space += direction
				distance += 1
		in_view[2] = 1./distance
		#if str(direction) == "[10  0]":
		#	print "---"
		#	print in_view[1]
		return in_view
	#
	def look(self):
		vision = []
		if self.direction == "RIGHT":
			# behind, left, straight, right
			look_directions=[[-10,0],[-10,-10],[0,-10],[10,-10],[10,0],[10,10],[0,10],[-10,10]]
		elif self.direction == "UP":
			look_directions=[[0,10],[-10,10],[-10,0],[-10,-10],[0,-10],[10,-10],[10,0],[10,10]]
		elif self.direction == "LEFT":
			look_directions=[[10,0],[10,10],[0,10],[-10,10],[-10,0],[-10,-10],[0,-10],[10,-10]]
		elif self.direction == "DOWN":
			look_directions=[[0,-10],[10,-10],[10,0],[10,10],[0,10],[-10,10],[-10,0],[-10,-10]]			
		for vector in look_directions:
			vector_look = self.lookInDirection(vector)
			for each in vector_look:
				vision.append(each)
		#print vision[9]
		return vision
	#
	def fitness(self):
		#"""
		if self.score < 10:
			fitness = ((self.life_time**2) * 2**self.score) 
		else:
			fitness = (self.life_time**2) * 2**10 * (self.score-9)
		#"""
		#fitness = self.life_time + (2**self.score)
		return fitness
	#
# ========== /Classes ==========



# ========== Functions ==========
def crossover(parentA, parentB):
	offspring = Snake()
	offspring.brain = deepcopy(parentA.brain)
	# Woot, brain surgery
	syn0 = offspring.brain.syn0
	syn1 = offspring.brain.syn1
	syn2 = offspring.brain.syn2
	# Split
	split0 = np.random.random()*np.size(syn0)
	split1 = np.random.random()*np.size(syn1)
	split2 = np.random.random()*np.size(syn2)
	full_size = split0 + split1 + split2
	counter = 0
	for i in range(len(syn0)):
		for j in range(len(syn0[i])):
			counter += 1
			if counter > full_size:
				syn0[i,j] = parentB.brain.syn0[i,j]
	#counter = 0
	for i in range(len(syn1)):
		for j in range(len(syn1[i])):
			counter += 1
			if counter > full_size:
				syn1[i,j] = parentB.brain.syn1[i,j]
	#counter = 0
	for i in range(len(syn2)):
		for j in range(len(syn2[i])):
			counter += 1
			if counter > full_size:
				syn2[i,j] = parentB.brain.syn2[i,j]
	# Cool, put it back together
	offspring.brain.syn0 = syn0
	offspring.brain.syn1 = syn1
	offspring.brain.syn2 = syn2
	return offspring

def drawSnake(snake):
	r,g,b = snake.color
	r = int(r)
	g = int(g)
	b = int(b)
	for pos in snake.getBody():
		pygame.draw.rect(window, pygame.Color(r,g,b), pygame.Rect(pos[0], pos[1], 10, 10))

def newPop(pop):
	# 1. Get best 4 snakes
	#	based on fitness
	global best_snake
	sorted_pop = []
	population = []
	score_list = []
	for snake in reversed(pop):
		score_list.append([snake, snake.fitness()])
	score_list.sort(key=lambda l: l[1], reverse=True)
	for each in score_list:
		sorted_pop.append(each[0])
	winners = sorted_pop[:int(snake_pop/4)] # Take the top 4 as is and add best_snake
	#population = winners
	population = []
	#
	b_snake = Snake()
	b_snake.brain = deepcopy(best_snake.brain)
	population.append(b_snake)
	#
	for each in winners:
		print each.id, "\t- ", int(each.fitness())
	# 2. build new pop with slight mutation score
	# add best snake
	if best_snake.fitness() < winners[0].fitness():
		best_snake = deepcopy(winners[0])
	print "Highscore: " + best_snake.id + ":" + str(best_snake.score)
	for each in population:
		each.reset()
	for x in xrange(snake_pop - int(snake_pop/4)):
		coin = np.random.choice([0,1])
		snake = Snake()
		parentA = np.random.choice(winners)
		parentB = np.random.choice(winners)
		parentR1 = np.random.choice(sorted_pop)
		parentR2 = np.random.choice(sorted_pop)
		if x < 5:
			if coin == 0:
				snake = crossover(winners[0], winners[1])
			elif coin == 1:
				snake = crossover(winners[1], winners[0])
		elif x < 50:
			if coin == 0:
				snake = crossover(parentA, parentB)
			elif coin == 1:
				snake = crossover(parentB, parentA)
		else:
			snake = crossover(parentR1, parentR2)
		snake.brain.mutate(.1)
		population.append(snake)
	# 3. return new pop
	return population

def gameOver():
	pygame.quit()
	sys.exit()
# ========== /Functions ==========



	


# ========== MAIN ==========
if __name__ == "__main__":
	window = pygame.display.set_mode((dimensions[0]+10,dimensions[1]+10))
	pygame.display.set_caption("WoW Snake!")
	fps = pygame.time.Clock()
	score = 0
	gen = 0
	world_age = 500
	draw_snakes = min(snake_pop, 1)
	#
	snakes = []
	for i in xrange(snake_pop):
		snake = Snake()
		snakes.append(snake)
	#
	best_snake = snakes[0]
	all_snakes = snakes[:]
	while True:
		world_age -= 1
		window.fill(pygame.Color(255,255,255))
		for snake in snakes:
			want_move = snake.brain.decision(snake.look())
			if nn_drive: 
				snake.nnCD(want_move)
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					gameOver()
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_RIGHT:
						snake.changeDirTo("RIGHT")
					if event.key == pygame.K_UP:
						snake.changeDirTo("UP")
					if event.key == pygame.K_DOWN:
						snake.changeDirTo("DOWN")
					if event.key == pygame.K_LEFT:
						snake.changeDirTo("LEFT")
			snake.move()
			if snake.id == snakes[0].id:
				drawSnake(snake)
			if snake.id == snakes[0].id:
				# Only draw the top level snake
				pygame.draw.rect(window, pygame.Color(255,0,0), pygame.Rect(snake.foodPos[0], snake.foodPos[1], 10, 10))
			if(snake.checkCollision() == 1):
				#gameOver()
				nil = snakes.pop(snakes.index(snake))
			if (len(snakes) < 1) or (world_age < 0):
				print "=== All Snakes Died ==="
				snakes = newPop(all_snakes)
				all_snakes = snakes[:]
				gen += 1
				print "Generation:", gen
				#world_age = 500 + 50 * gen
				world_age = 10000
		pygame.display.set_caption("WoW Snake | Generation: " + str(gen) + " | Time till Doom: " + str(world_age))
		pygame.display.flip()
		fps.tick(frame_rate)
