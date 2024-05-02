import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
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
		total_degree = sum(sum(node.connections) for node in self.nodes)
		return total_degree / len(self.nodes)

	def get_mean_clustering(self):
		#Your code for task 3 goes here
		#Calculate the mean clustering
		total_clustering = 0
		for node in self.nodes:
			node_neighbors = [self.nodes[i] for i, is_connected in enumerate(node.connections) if is_connected]
			num_neighbors = len(node_neighbors)
			num_possible_connections = num_neighbors * (num_neighbors - 1) / 2

			if num_possible_connections > 0:
				num_actual_connections = 0
				for i in range(num_neighbors):
					for j in range(i + 1, num_neighbors):
						neighbor1 = node_neighbors[i]
						neighbor2 = node_neighbors[j]
						if node.connections[neighbor1.index] == 1 and node.connections[neighbor2.index] == 1:
							if neighbor1.connections[neighbor2.index] == 1 and neighbor2.connections[neighbor1.index] == 1:
								num_actual_connections += 1
				node_clustering = num_actual_connections / num_possible_connections
			else:
				node_clustering = 0

			total_clustering += node_clustering

		return total_clustering / len(self.nodes)

	def get_mean_path_length(self):
		#Your code for task 3 goes here
		total_path_length = 0
		for node in self.nodes:
			distances = self.bfs(node)
			node_path_length = sum(distances.values()) / (len(self.nodes) - 1)
			total_path_length += node_path_length
		return total_path_length / len(self.nodes)

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

	def bfs(self, start):
		visited = {node: False for node in self.nodes}
		distances = {node: 0 for node in self.nodes}
		queue = [start]
		visited[start] = True
		while queue:
			current = queue.pop(0)
			for neighbour_index, is_connected in enumerate(current.connections):
				if is_connected and not visited[self.nodes[neighbour_index]]:
					neighbour = self.nodes[neighbour_index]
					visited[neighbour_index] = True
					distances[neighbour_index] = distances[current] + 1
					queue.append(neighbour)
		return distances

	def plot_network(self):
		plt.figure(figsize=(8, 8))

		# Generate random positions for the nodes
		positions = np.random.rand(len(self.nodes), 2)

		for i, node in enumerate(self.nodes):
			x, y = positions[i]
			plt.scatter(x, y, color=cm.viridis(node.value), s=500)
			plt.text(x, y, str(node.index), ha='center', va='center', fontsize=10)

		for i, node in enumerate(self.nodes):
			for neighbour_index, is_connected in enumerate(node.connections):
				if is_connected:
					x1, y1 = positions[i]
					x2, y2 = positions[neighbour_index]
					plt.plot([x1, x2], [y1, y2], color='black', linewidth=1)

		plt.title("Networks")
		plt.axis('off')
		plt.show()

	def make_ring_network(self, N, neighbour_range=1):
		"""
		Makes a ring network following a similar structure to make random network.
		Neighbour range default value set to 1.
		:param N: Size of network (number of nodes)
		:param neighbour_range: Range either side of node which a connection if formed.
		:return: Returns ring network.
		"""

		# creates a network of size n, follows same structure as make random network
		for node_number in range(N):
			value = np.random.random()
			# sets initial value of connections to 0
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
		"""
		Function makes small world network of size N and re wires connection at a default probability of 0.2.
		:param N: Size of Network (number of nodes)
		:param re_wire_prob: Probability a connection is rewired
		:return: Returns small world network.
		"""
		# creates a base ring network

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

						# Establishes the new connection
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
	assert(network.get_mean_clustering()==0), network.get_mean_clustering()
	assert(network.get_mean_path_length()==2.777777777777778), network.get_mean_path_length()

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
	assert(network.get_mean_clustering()==0),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==5), network.get_mean_path_length()

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
	assert(network.get_mean_clustering()==1),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==1), network.get_mean_path_length()

	print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''
def get_neighbours_opinions(population=None, row=None, col=None, node=None, network=None, Number_of_nodes=None):
	'''
	Retrieves the value of the neighbours surrounding the
	randomly picked value in the numpy array and putting
	them into a list
	The modulo symbols are neccessary as to wrap the
	numbers round once it reaches a border
	'''
	neighbours_values = []
	if population is not None:
		n, m = population.shape
		neighbours_values.append(population[(row - 1), col])
		neighbours_values.append(population[(row + 1) % n, col])
		neighbours_values.append(population[row, col - 1])
		neighbours_values.append(population[row, (col + 1) % m])

	if network is not None:
		neighbours = []
		# iterates through all nodes
		for neighbour in range(Number_of_nodes):

			if network.nodes[node].connections[neighbour] == 1:
				neighbours_values.append(network.nodes[neighbour].value)

	return neighbours_values

def calculate_agreement(population=None, row=None, col=None, network=None, external=0.0, node=None, Number_of_nodes=None):
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
	if population is not None:
		individual_opinion = population[row, col]
		#  Uses function to gain and store the neighbours opinions in a variable
		neighbour_opinion = get_neighbours_opinions(population=population, row=row, col=col)

	if network is not None:
		individual_opinion = network.nodes[node].value
		neighbour_opinion = get_neighbours_opinions(network=network, node=node, Number_of_nodes=Number_of_nodes)

	for opinion in neighbour_opinion:
		agreement += individual_opinion * opinion

	agreement += external * individual_opinion  # enforcing original opinion of individual

	return agreement


def ising_step(population=None, network=None, external=0.0, alpha=1.0, Number_of_nodes=None):
	'''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''
	if population is not None:
		n_rows, n_cols = population.shape
		row = np.random.randint(0, n_rows)
		col = np.random.randint(0, n_cols)

		agreement = calculate_agreement(population=population, row=row, col=col, external=0.0)

		if agreement < 0:
			population[row, col] *= -1
		elif alpha:
			random_num = random.random()
			p = math.e ** (-agreement / alpha)  # chance that it'll flip anyways
			if random_num < p:
				population[row, col] *= -1

	if network is not None:
		# choose node to check +1 to allow for last node to be checked
		node = np.random.randint(0, Number_of_nodes)

		agreement = calculate_agreement(network=network, node=node, Number_of_nodes=Number_of_nodes)

		if agreement < 0:
			network.nodes[node].value *= -1
		elif alpha:
			random_num = random.random()
			p = math.e ** (-agreement / alpha)  # chance that it'll flip anyways
			if random_num < p:
				network.nodes[node].value *= -1


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


def ising_main(population=None, network=None, alpha=None, external=0.0, Number_of_nodes=None):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_axis_off()

	if population is not None:

		im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

		# Iterating an update 100 times
		for frame in range(100):
			# Iterating single steps 1000 times to form an update
			for step in range(1000):
				ising_step(population=population, external=external, alpha=alpha)
			print('Step:', frame, end='\r')
			plot_ising(im, population)

	if network is not None:
		# Iterating an update 100 times
		for frame in range(100):
			# Iterating single steps 1000 times to form an update
			for step in range(1000):
				ising_step(network=network, external=external, alpha=alpha, Number_of_nodes=Number_of_nodes)
			print('Step:', frame, end='\r')


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

# initialize opinions of each individuals with random values between 0 and 1.
def initialize_opinions(num_individuals):
	return np.random.rand(num_individuals)


# update opinions.
def update_opinion(opinions, T, beta):
	num_individuals = len(opinions)
	i = np.random.randint(num_individuals)
	neighbor = select_neighbor(i, num_individuals)
	diff = abs(opinions[i] - opinions[neighbor])
	if diff < T:
		opinions[i] += beta * (opinions[neighbor] - opinions[i])
		opinions[neighbor] += beta * (opinions[i] - opinions[neighbor])


# randomly choose left or right neighbor.
def select_neighbor(i, num_individuals):
	return (i - 1) % num_individuals if np.random.rand() < 0.5 else (i + 1) % num_individuals


# run opinion updates and record history.
def updates(num_individuals, T, beta, num_updates):
	opinions = initialize_opinions(num_individuals)
	update_history = np.zeros((num_updates, num_individuals))
	for update_step in range(num_updates):
		update_opinion(opinions, T, beta)
		update_history[update_step] = opinions.copy()
	return update_history


# plot opinion distribution.
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


# test the model under different parameters.
def test_defuant():
	num_individuals = 100
	T_defuant = 0.2
	T_big = 0.8
	beta_defuant = 0.2
	beta_big = 0.8
	num_updates = 10000

	# test with defuant beta and defuant T
	update_history_defuant = updates(num_individuals, T_defuant, beta_defuant, num_updates)
	final_opinions_defuant = update_history_defuant[-1]
	assert len(set(final_opinions_defuant)) > 1, "Opinions did not start to diverge into clusters with default beta"

	# test with small beta
	update_history_small = updates(num_individuals, T_defuant, beta_defuant, num_updates)
	final_opinions_small = update_history_small[-1]

	# test with big beta
	update_history_big = updates(num_individuals, T_defuant, beta_big, num_updates)
	final_opinions_big = update_history_big[-1]
	assert len(set(final_opinions_big)) < len(set(final_opinions_small)), "Opinions did not converge faster with bigger beta"

	print("All tests passed successfully!")


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
	"""
	Main function to handle arguments passed through the terminal.
	:return:
	"""

	parser = argparse.ArgumentParser()
	#Flag definition
	parser.add_argument("-ising_model", action='store_true')
	parser.add_argument("-external", type=float, default=0)
	parser.add_argument("-alpha", type=float, default=1)
	parser.add_argument("-test_ising", action='store_true')
	parser.add_argument("-use_network", nargs= 1, type= int)
	parser.add_argument("-ring_network", nargs=1, type=int)
	parser.add_argument("-small_world", nargs=1, type=int)
	parser.add_argument("-re_wire", type=float, default=0.2)
	parser.add_argument("-defuant", action='store_true')
	parser.add_argument("-beta", type=float, nargs=1)
	parser.add_argument("-threshold", type=float, nargs=1)
	parser.add_argument("-test_defuant", action= 'store_true')
	parser.add_argument("-network",nargs= 1, type=int)
	parser.add_argument("-test_network", action="store_true")

	#Variable definition
	args = parser.parse_args()
	external = args.external
	alpha = args.alpha
	N_ring = args.ring_network
	N_small = args.small_world
	re_wire_prob = args.re_wire
	defuant = args.defuant
	beta_command = args.beta
	threshold_command = args.threshold
	test_def = args.test_defuant
	network_command = args.network
	test_net = args.test_network
	ising_network = args.use_network

	# When ising_model called
	if args.ising_model:

		# if using a network
		if ising_network:
			#
			ising_network = ising_network[0]
			re_wire_prob_ising = 0.2
			# creates network
			network = Network()
			# creates small world network
			network.make_small_world_network(ising_network, re_wire_prob_ising)
			# iterates over amount of nodes
			for node in range(ising_network):
				# finds opinion of node
				random_opinion = np.random.choice([-1, 1])
				# appends opinion to node
				network.nodes[node].value = random_opinion
				ising_main(network=network, alpha=alpha, external=external, Number_of_nodes=ising_network)

		# if using normal ising model
		else:
			pop = np.random.choice([-1, 1], size=(100, 100))
			ising_main(population=pop, alpha=alpha, external=external)

	# tests ising model
	if args.test_ising:
		test_ising()

	# for the ring network
	if N_ring:
		# converts passed through value into an integer rather than a list value
		N_ring = N_ring[0]
		# Create a Network instance
		network = Network()
		#creates ring network
		network.make_ring_network(N_ring)
		# Plot the network
		network.plot()
		# Show the plot
		plt.show()

	# if small world model called
	if N_small:
		# coverts size of model into integer value
		N_small = N_small[0]
		# if re wire probability provided it converts to integer
		if type(re_wire_prob) == list:
			re_wire_prob = re_wire_prob[0]
		# Create a Network instance
		network = Network()
		#creates small world network
		network.make_small_world_network(N_small, re_wire_prob)
		# Plot the network
		network.plot()
		# Show the plot
		plt.show()

	if defuant:
		num_individuals = 100
		T = 0.2
		beta = 0.2
		num_updates = 10000

		# check if command line arguments are provided
		if beta_command:
			beta = beta_command[0]
		if threshold_command:
			T = threshold_command[0]

		update_history = updates(num_individuals, T, beta, num_updates)
		plot_histogram(update_history[-1])
		plot_updates(update_history)
	if test_def:
		test_defuant()

	if network_command:
		connection_probability = 0.5
		num_nodes = network_command
		network = Network(num_nodes)
		network.make_random_network(num_nodes, connection_probability)
		network.plot_network()
		plt.show()

	if test_net:
		test_networks()

if __name__=="__main__":
	main()
