import numpy as np
import matplotlib.pyplot as plt
import sys

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
        print(diff,T)
        opinions[i] += beta * (opinions[neighbor] - opinions[i])
        opinions[neighbor] += beta * (opinions[i] - opinions[neighbor])


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


def main():
    import sys
    if "-test_defuant" in sys.argv:
        test_defuant()
    else:
        num_individuals = 100
        T = 0.1
        beta = 0.5
        num_updates = 10000

        update_history = updates(num_individuals, T, beta, num_updates)
        plot_histogram(update_history[-1])
        plot_updates(update_history)


if __name__ == "__main__":
    main()




