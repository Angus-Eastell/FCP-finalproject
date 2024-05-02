# FCP-finalproject
git for final fcp project
Task2
Ps：When the first histogram is generated, you need to turn it off so that the second scatter chart will be displayed.

Import the required libraries, numpy, matplotlib.
 
In the terminal, the command “-defuant” is used to run the defuant model，“-test_defuant” is used to run the test code of the model，“-beta” is used to assign a value to the beta，“-threshold” is used to assign a value to the T.
                      
First of all, to give each individual an initial value of an opinion, I created a function called initialise_opinions, using the numpy.random.rand() function in the numpylibrary to randomly generate an opinion for each individual. Initial value. In this way, an array of random numbers is generated, and the length of this array depends on num_individuals, indicating the total number of people in the model. Here, I set its value to 100.
 
Then, I created a function called update_opinion（）to update the individual's opinion according to the calculation formula provided by the task. In this process, I created an index i to indicate the individuals selected in the current update to change their opinion values and make the code clearer. Randomly select an individual and then randomly select the neighbor on the left or right of the individual, and update the individual's views according to the differences between the selected individual and the neighbour's views.
 
After that, I created a select_neighbor（）function to complete the process of randomly selecting one from the left and right neighbors of the selected individual required in the update_opinion（）function.
 
Finally, I created a function called updates, which is different from the previous update_opinion（）function. The update function is used to run the process of randomly selecting individuals, comparing with the opinion values of neighbors, and updating opinions many times. In this function, I first generated an array of opinions containing the random initial opinions of all individuals. After that, I created a two-dimensional array through the numpy.zeros function in the numpy library to record the changes of individual opinions after each update. Each row in the matrix represents the number of updates, and each column represents an individual's opinion. That is to say, column m of line n represents the opinion of the mth individual after the nth update. After that, I used the for loop iteration "num_updates", and the number of iterations was set to 10000 to indicate multiple opinion updates, and copied the currently updated individual opinion to the corresponding position in the update_history array to record each History of updated opinions. Finally, the update_history array that records individual opinions after each update is returned to the calling function.
 
In the drawing part, the plot_histogram（）function is used to draw the histogram of opinion distribution, while the plot_updates（）function is used to draw the evolutionary scatter map of individual opinions with the number of updates.
 
In the test code part, according to the requirements of the question, I created a function called test_defuant（）to test if the update of individual opinions in the whole simulation is right. First, the parameters used in the test are set, and then the smaller and larger beta values are used to simulate respectively. After that, I conducted assertive tests to test the use of the default beta and T values, and the larger beta and T values, respectively, and compared the default values of the two as the smaller values with the larger values of the two. and checked whether the results of the simulation of the two situations met expectations respectively to verify the correctness of my simulation process.
 
Finally, I used an if statement to check whether the execution parameters of the test code appear in the command line parameters. If so, run the test code. If not, run the main code according to the set parameters, or according to the command "- Beta" and command "- Threshold" . This is, the simulated program for task2.
