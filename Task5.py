# Task 5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys
import random
import math


# Change the get neighbours function to work with a network
# Integrate the code within the existing task 1 to work when a network when the class is called
# Then use if


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
        # Your code for task 4 goes here

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
        ax.set_xlim([-1.1 * network_radius, 1.1 * network_radius])
        ax.set_ylim([-1.1 * network_radius, 1.1 * network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3 * num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i + 1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')


"""
=================================================
This section is for Task 5
=================================================

"""


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


def calculate_agreement(population=None, row=None, col=None, network=None, external=0.0, node=None,
                        Number_of_nodes=None):
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
        neighbour_opinion = get_neighbours_opinions(population=population, row=row,
                                                    col=col)  # Uses function to gain and store the neighbours opinions in a variable

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


def main():
    parser = argparse.ArgumentParser()

    # Flag definition
    parser.add_argument("-ising_model", action='store_true')
    parser.add_argument("-external", type=float, default=0)
    parser.add_argument("-alpha", type=float, default=1)
    parser.add_argument("-test_ising", action='store_true')
    parser.add_argument("-use_network", action='store_true')

    # Variable definition
    args = parser.parse_args()
    external = args.external
    alpha = args.alpha
    # You should write some code for handling flags here
    if args.ising_model:
        if args.use_network:
            # values
            N = 10
            re_wire_prob = 0.2
            # creates network
            network = Network()
            # creates small world network
            network.make_small_world_network(N, re_wire_prob)
            # iterates over amount of nodes
            for node in range(10):
                # finds opinion of node
                random_opinion = np.random.choice([-1, 1])
                # appends opinion to node
                network.nodes[node].value = random_opinion
                ising_main(network=network, alpha=alpha, external=external, Number_of_nodes=N)

        else:
            pop = np.random.choice([-1, 1], size=(100, 100))
            ising_main(population=pop, alpha=alpha, external=external)


if __name__ == "__main__":
    main()