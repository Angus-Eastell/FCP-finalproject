import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys
import random
import math

class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value

class Network: 

	def __init__(self, nodes=None):

		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes 

	def get_mean_degree(self):
		#Your code  for task 3 goes here
		#Calculate the mean degree of the network
		total_degree = sum(len(node.neighbors) for node in self.nodes)
		return total_degree / len(self.nodes)

	def get_mean_clustering(self):
		#Your code for task 3 goes here
		#Calculate the mean clustering
		total_path_length = 0
		for node in self.nodes:
			paths = self.shortest_path_length(node.id)
			node_path_length = sum(paths.values()) / (len(self.nodes) - 1)
			total_path_length += node_path_length
		return total_path_length / len(self.nodes)

	def get_mean_path_length(self):
		#Your code for task 3 goes here
		total_clustering = 0
		for node in self.nodes:
			num_neighbors = len(node.neighbors)
			num_possible_connections = num_neighbors * (num_neighbors - 1) / 2
			if num_possible_connections > 0:
				num_actual_connections = 0
				for neighbor1 in node.neighbors:
					for neighbor2 in node.neighbors:
						if neighbor1 in neighbor2.neighbors:
							num_actual_connections += 1
				node_clustering = num_actual_connections / num_possible_connections
			else:
				node_clustering = 0
			total_clustering += node_clustering
		return total_clustering / len(self.nodes)

	def make_random_network(self, N, connection_probability=0.5):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	def make_ring_network(self, N, neighbour_range=1):
		# Your code  for task 4 goes here

		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		# Connect nodes in a ring
		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index - neighbour_range, index + neighbour_range + 1):
				# prevents self connections
				if neighbour_index != index:
					# Modulus to allow for edge nodes
					neighbour_index = neighbour_index % N
					# prevents repeat connections
					if node.connections[neighbour_index] == 0:
						node.connections[neighbour_index] = 1
						self.nodes[neighbour_index].connections[index] = 1


	def make_small_world_network(self, N, re_wire_prob=0.2):
		#Your code for task 4 goes here

		self.make_ring_network(N, 2)

		# Iterate over every node
		for node in self.nodes:
			# Iterate through connections of the node, tracking the neighbour index
			for neighbour_index, connection in enumerate(node.connections):
				# Ensure no self-connection and check if a connection exists to be rewired
				if neighbour_index != node.index and connection == 1:

					# Check the rewiring probability
					if random.random() < re_wire_prob:

						# Set the previous connection to zero
						node.connections[neighbour_index] = 0
						self.nodes[neighbour_index].connections[node.index] = 0

						# Select a random neighbour to switch to
						new_index = node.index
						while new_index == node.index or node.connections[new_index] == 1:
							new_index = random.randint(0, N - 1)

						# Establish the new connection
						node.connections[new_index] = 1
						self.nodes[new_index].connections[node.index] = 1


	def plot(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''
def get_neighbours_opinions(population, row, col):
		'''
		Retrieves the value of the neighbours surrounding the 
		randomly picked value in the numpy array and putting 
		them into a list
		The modulo symbols are neccessary as to wrap the
		numbers round once it reaches a border
		'''
		n,m = population.shape
		neighbours = []
		neighbours.append(population[(row-1), col])
		neighbours.append(population[(row+1)%n, col])
		neighbours.append(population[row, col-1])
		neighbours.append(population[row, (col+1)%m])
		return neighbours

def calculate_agreement(population, row, col, external=0.0):
	'''
	This function should return the extent to which a cell agrees with its neighbours.
	Inputs: population (numpy array)
			row (int)
			col (int)
			external (float)
	Returns:
			change_in_agreement (float)
	'''
	agreement = 0
	individual_opinion = population[row, col]
	neighbour_opinion = get_neighbours_opinions(population, row, col) #Uses function to gain and store the neighbours opinions in a variable
	for opinion in neighbour_opinion:
		agreement += individual_opinion * opinion
	agreement += external * individual_opinion #enforcing original opinion of individual
	return agreement

def ising_step(population, external=0.0, alpha=1.0):
	'''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''
	
	n_rows, n_cols = population.shape
	row = np.random.randint(0, n_rows)
	col = np.random.randint(0, n_cols)

	agreement = calculate_agreement(population, row, col, external=0.0)

	if agreement < 0:
		population[row, col] *= -1
	elif alpha:
		random_num = random.random()
		p = math.e**(-agreement/alpha) #chance that it'll flip anyways
		if random_num < p:
			population[row, col] *= -1


def plot_ising(im, population):
	'''
	This function will display a plot of the Ising model
	'''

	new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
	im.set_data(new_im)
	plt.pause(0.1)

def test_ising():
	'''
	This function will test the calculate_agreement function in the Ising model
	'''

	print("Testing ising model calculations")
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1)==4), "Test 1"

	population[1, 1] = 1.
	assert(calculate_agreement(population,1,1)==-4), "Test 2"

	population[0, 1] = 1.
	assert(calculate_agreement(population,1,1)==-2), "Test 3"

	population[1, 0] = 1.
	assert(calculate_agreement(population,1,1)==0), "Test 4"

	population[2, 1] = 1.
	assert(calculate_agreement(population,1,1)==2), "Test 5"

	population[1, 2] = 1.
	assert(calculate_agreement(population,1,1)==4), "Test 6"

	"Testing external pull"
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1,1)==3), "Test 7"
	assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
	assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
	assert(calculate_agreement(population,1,1, -10)==14), "Test 10"

	print("Tests passed")


def ising_main(population, alpha=None, external=0.0):

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_axis_off()
	im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

	# Iterating an update 100 times
	for frame in range(100):
		# Iterating single steps 1000 times to form an update
		for step in range(1000):
			ising_step(population, external, alpha)
		print('Step:', frame, end='\r')
		plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

# initialize opinions of individuals with random values between 0 and 1.
def initialize_opinions(num_individuals):
	return np.random.rand(num_individuals)


# update opinions.
def update_opinion(opinions, T, beta):
	num_individuals = len(opinions)
	i = np.random.randint(num_individuals)
	neighbor = select_neighbor(i, num_individuals)
	diff = abs(opinions[i] - opinions[neighbor])
	if diff < T:
		mean_opinion = (opinions[i] + opinions[neighbor]) / 2
		opinions[i] += beta * (mean_opinion - opinions[i])
		opinions[neighbor] += beta * (mean_opinion - opinions[neighbor])


# randomly choose left or right neighbor.
def select_neighbor(index, num_individuals):
	return (index - 1) % num_individuals if np.random.rand() < 0.5 else (index + 1) % num_individuals


# run opinion updates and record history.
def updates(num_individuals, T, beta, num_updates):
	opinions = initialize_opinions(num_individuals)
	update_history = np.zeros((num_updates, num_individuals))
	for update_step in range(num_updates):
		update_opinion(opinions, T, beta)
		update_history[update_step] = opinions.copy()
	return update_history


# Plot opinion distribution.
def plot_histogram(opinions):
	plt.figure(figsize=(10, 5))
	plt.hist(opinions, bins=20, alpha=0.75)
	plt.title('Opinion Distribution')
	plt.xlabel('Opinion')
	plt.ylabel('Frequency')
	plt.show()


# plot opinion evolution over time.
def plot_updates(update_history):
	plt.figure(figsize=(15, 8))
	num_updates, num_individuals = update_history.shape
	for person in range(num_individuals):
		plt.scatter(np.arange(num_updates), update_history[:, person], color='red')
	plt.title('Opinion Dynamics')
	plt.xlabel('Time Step')
	plt.ylabel('Opinion')
	plt.ylim(0, 1)
	plt.show()


# Test the model under different parameters.
def test_defuant():
	num_individuals = 10
	T = 0.5
	beta_small = 0.1
	beta_big = 0.9
	num_updates = 100

	update_history_small = updates(num_individuals, T, beta_small, num_updates)
	assert update_history_small.shape == (num_updates, num_individuals), "Incorrect shape"
	final_opinions = update_history_small[-1]
	assert (final_opinions.max() - final_opinions.min()) < T, "No convergence"

	update_history_big = updates(num_individuals, T, beta_big, num_updates)
	final_opinions = update_history_big[-1]
	assert (final_opinions.max() - final_opinions.min()) < T, "No convergence"


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
	parser = argparse.ArgumentParser()
	
	#Flag definition
	parser.add_argument("-ising_model", action='store_true')
	parser.add_argument("-external", type=float, default=0)
	parser.add_argument("-alpha", type=float, default=1)
	parser.add_argument("-test_ising", action='store_true')
	
	#Variable definition
	args = parser.parse_args()
	external = args.external
	alpha = args.alpha
	#You should write some code for handling flags here
	if args.ising_model:
		pop = np.random.choice([-1,1],size=(100,100))
		ising_main(pop, alpha, external)


	# small world

	# Create Parser object
	# Add arguments
	parser.add_argument("-ring_network", nargs=1, type=int, default=10)
	parser.add_argument("-small_world", nargs=1, type=int, default=10)
	parser.add_argument("-re_wire", type=float, default=0.2)  #

	# Variable definition
	args = parser.parse_args()
	N_ring = args.ring_network
	N_small = args.small_world
	re_wire_prob = args.re_wire
	if type(N_ring) == list:
		N_ring = N_ring[0]

	if type(N_small) == list:
		N_small = N_small[0]

	if type(re_wire_prob) == list:
		re_wire_prob = re_wire_prob[0]

	if N_ring:
		# Create a Network instance
		network = Network()
		network.make_ring_network(N_ring)
		# Plot the network
		network.plot()
		# Show the plot
		plt.show()

	if N_small:
		# Create a Network instance
		network = Network()
		network.make_small_world_network(N_small, re_wire_prob)
		# Plot the network
		network.plot()
		# Show the plot
		plt.show()

	if "-test_defuant" in sys.argv:
		test_defuant()
	else:
		num_individuals = 100
		T = 0.2
		beta = 0.2
		num_updates = 10000

		update_history = updates(num_individuals, T, beta, num_updates)
		plot_histogram(update_history[-1])
		plot_updates(update_history)

if __name__=="__main__":
	main()
