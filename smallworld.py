import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

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

    def make_random_network(self, N, connection_probability):
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
            for neighbour_index in range(index + 1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_ring_network(self, N, neighbour_range=2):
        # Your code  for task 4 goes here
        self.nodes = []
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
        # Create a small network
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


#Create Parser object
parser = argparse.ArgumentParser()
#Add arguments
parser.add_argument("-ring_network", nargs=1, type=int, default= 10)
parser.add_argument("-small_world", nargs=1, type=int, default= 10)
parser.add_argument("-re_wire", type=float, default= 0.2)#

#Process the provided data
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



