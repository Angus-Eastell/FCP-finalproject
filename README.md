# FCP-finalproject
Read me for final project:

Import the required libraries, numpy, matplotlib, argparse, math, random.

Task 1:
To run task one, use the “-ising_model” flag; this will cause an animation to play representing the Ising model. To alter the output of the figure to show different representations you can alter the external and alpha values of the model. Hence affecting the way each other's opinions affect their surroundings. This can be inputted by giving a value for each by adding “external = x” or “alpha = x” where x is a positive number.

Task 2:
Ps：When the first histogram is generated, you need to turn it off so that the second scatter chart will be displayed.
 
In the terminal, the command “-defuant” is used to run the defuant model，“-test_defuant” is used to run the test code of the model，“-beta” is used to assign a value to the beta，“-threshold” is used to assign a value to the T.

First of all, to give each individual an initial value of an opinion, I created a function called initialise_opinions, using the numpy.random.rand() function in the numpylibrary to randomly generate an opinion for each individual. Initial value. In this way, an array of random numbers is generated, and the length of this array depends on num_individuals, indicating the total number of people in the model. Here, I set its value to 100.
 
Then, I created a function called update_opinion（）to update the individual's opinion according to the calculation formula provided by the task. In this process, I created an index i to indicate the individuals selected in the current update to change their opinion values and make the code clearer. Randomly select an individual and then randomly select the neighbor on the left or right of the individual, and update the individual's views according to the differences between the selected individual and the neighbour's views.
 
After that, I created a select_neighbor（）function to complete the process of randomly selecting one from the left and right neighbors of the selected individual required in the update_opinion（）function.
 
Finally, I created a function called updates, which is different from the previous update_opinion（）function. The update function is used to run the process of randomly selecting individuals, comparing with the opinion values of neighbors, and updating opinions many times. In this function, I first generated an array of opinions containing the random initial opinions of all individuals. After that, I created a two-dimensional array through the numpy.zeros function in the numpy library to record the changes of individual opinions after each update. Each row in the matrix represents the number of updates, and each column represents an individual's opinion. That is to say, column m of line n represents the opinion of the mth individual after the nth update. After that, I used the for loop iteration "num_updates", and the number of iterations was set to 10000 to indicate multiple opinion updates, and copied the currently updated individual opinion to the corresponding position in the update_history array to record each History of updated opinions. Finally, the update_history array that records individual opinions after each update is returned to the calling function.
 
In the drawing part, the plot_histogram（）function is used to draw the histogram of opinion distribution, while the plot_updates（）function is used to draw the evolutionary scatter map of individual opinions with the number of updates.
 
In the test code part, according to the requirements of the question, I created a function called test_defuant（）to test if the update of individual opinions in the whole simulation is right. First, the parameters used in the test are set, and then the smaller and larger beta values are used to simulate respectively. After that, I conducted assertive tests to test the use of the default beta and T values, and the larger beta and T values, respectively, and compared the default values of the two as the smaller values with the larger values of the two. and checked whether the results of the simulation of the two situations met expectations respectively to verify the correctness of my simulation process.
 
Finally, I used an if statement to check whether the execution parameters of the test code appear in the command line parameters. If so, run the test code. If not, run the main code according to the set parameters, or according to the command "- Beta" and command "- Threshold" . This is, the simulated program for task2.

Task 3:
This network analysis contains Python code for creating and analysing various types of networks, including random networks. 
The code provides functionality to calculate the mean degree, mean clustering coefficient, and mean path length of the networks, as well as visualize the network structure. 
Furthermore, this will create a random network of the specified size and plot the network structure. It will also print the mean degree, mean clustering coefficient, and mean path length to the console with the appropriate command-line arguments. 
For the -test_network, This will run a set of tests to verify the correctness of the get_mean_degree(), get_mean_clustering, and get_mean_path_length functions. To make networks with random connectivity the get_mean_degree calculates the mean degree of the network, get_mean_clustering calculates the mean clustering coefficient of the network, get_mean_path_length calculates the mean path length of the network using a breadth-first search (BFS) algorithm and make_random_network(N, connection_probability=0.5) creates a random network with N nodes and a given connection probability. 
The bfs(self, start) performs a BFS traversal starting from the given node, returning a dictionary of distances from the start node to all other nodes then plot_network visualises the network structure using Matplotlib. I created and plotted a network of size N by calling the program with a flag -network. The code should print the mean degree, average path length, and clustering coefficient for the network to the terminal by a random network of size 10. Finally,  -test_network should run the test functions that I have provided. 


Task 4:
For the make_ring_network function I created a network of nodes initially with their connections set to 0. Then using the modulo function I assigned connections to nearby nodes based on the distace given in the function. This is then plotted when called in the main function with -ring_network and then a integer value for the size of the network.
For the small world network, it is neccessary to first call the make_ring_function with neighbour range 2. I then iterated through the nodes and for each connection tested it to see if it would be rewired. During this if statements were used to ensure no self-connections. This was again plotted in the main function with -small_world with an integer fo the size of the network and an optional -re_wire argument which can be followed by a float to determine the re wire probability used. If no re wire probability is provided it is set to the defualt 0.2. 

Task 5:

We chose to implement the small world network into the ising model. When called using the -use_network argument after the -ising_model argument. The small world network is then created in the main function and assinged a random opinion. Every ising model function is edited to allow for the passing through of a network. Each function works the same however is adjusted for the needs of a network instead of an array. This was then attempted to be plotted in the form of an animation, however we were unable to collate them all into one figure. The average of the opinion is also calculated at every time period and then is plotted at the end.
