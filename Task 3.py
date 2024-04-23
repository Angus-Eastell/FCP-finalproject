import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Node:

    def __init__(self, value, number, connections=None):

        self.value = value #Store a value for the node
        self.index = number #Store the index of the node
        self.connections = connections #List to store the neighboring nodes

class Network:

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = [] #Initialise an empty list of nodes
        else:
            self.nodes = nodes #Use the provided list of nodes

    def get_mean_degree(self):
        #Calculate the mean degree of the network
        total_degree = sum(len(node.neighbors) for node in self.nodes)
        return total_degree / len(self.nodes)

    def get_mean_clustering(self):
        #Calculate the mean clustering
        total_path_length = 0
        for node in self.nodes:
            paths = self.shortest_path_length(node.id)
            node_path_length = sum(paths.values()) / (len(self.nodes) - 1)
            total_path_length += node_path_length
        return total_path_length / len(self.nodes)


    def get_mean_path_length(self):
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


    def make_random_network(self, N, connection_probability):
        #Create a random network with N nodes and the specified connection probability
        self.nodes = []
        for node_number in range(N):
            value = np.random.random() #Assign a random value to the node
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

            for (index, node) in enumerate(self.nodes):
                for neighbour_index in range(index + 1, N):
                    if np.random.random() < connection_probability:
                        node.connections[neighbour_index] = 1
                        self.nodes[neighbour_index].connections[index] = 1

def plot(self):
    #Plot the network in a circular layout
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
    assert (network.get_mean_degree() == 2), network.get_mean_degree()
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 2.777777777777778), network.get_path_length()

    #One-sided network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert (network.get_mean_degree() == 1), network.get_mean_degree()
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 5), network.get_path_length()

    #Fully connected network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert (network.get_mean_degree() == num_nodes - 1), network.get_mean_degree()
    assert (network.get_clustering() == 1), network.get_clustering()
    assert (network.get_path_length() == 1), network.get_path_length()

    print("All tests passed")