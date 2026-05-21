import numpy as np
import matplotlib.pyplot as plt

# HMM Parameters
def generate_hmm_params(num_states = 3, obs_dim = 2):

    # Initial state, uniform
    pi = np.ones(num_states)/num_states

    # Transition, sticky with rows summing to one
    A = np.full((num_states,num_states), .05 / (num_states-1))
    np.fill_diagonal(A, .95)

    # Emissions
    mus = np.array([
        [2, 2],
        [-2,2],
        [0,-2]
    ])

    sigmas = np.stack([np.eye(obs_dim) * .5 for _ in range(num_states)]) # shape: (num_states, obs_dim, obs_dim)

    return pi, A, mus, sigmas

# Sequence
def generate_sequence(pi, A, mus, sigmas, T=100):

    num_states = len(pi)
    obs_dim = mus.shape[1]

    states = np.zeros(T, dtype = int)
    observations = np.zeros((T, obs_dim))

    states[0] = np.random.choice(num_states, p=pi)
    observations[0] = np.random.multivariate_normal(mus[states[0]], sigmas[states[0]])

    for t in range(1,T):
        states[t] = np.random.choice(num_states, p=A[states[t-1]])
        observations[t] = np.random.multivariate_normal(mus[states[t]], sigmas[states[t]])

    return observations, states


def generate_dataset(num_sequences=50, T=100, obs_dim = 2, num_states=3):

    pi, A, mus, sigmas = generate_hmm_params(num_states, obs_dim)

    all_observations = []
    all_states = []

    for _ in range(num_sequences):
        obs, states = generate_sequence(pi, A, mus, sigmas, T)
        all_observations.append(obs)
        all_states.append(states)

    return np.array(all_observations), np.array(all_states), {'pi':pi, 'A':A, 'mus':mus, 'sigmas':sigmas}


def visualize_sequence(observations, states, num_states=3):
    """Plot a single sequence - observations colored by true hidden state."""
    colors = ['red', 'blue', 'green']
    T = len(states)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: 2D scatter of observations colored by state
    for k in range(num_states):
        mask = states == k
        axes[0].scatter(observations[mask, 0], observations[mask, 1],
                       c=colors[k], label=f'State {k}', alpha=0.6)
    axes[0].set_title('Observations colored by true state')
    axes[0].legend()

    # Right: time series of state sequence
    axes[1].plot(range(T), states)
    axes[1].set_title('Hidden state sequence over time')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('state')

    plt.tight_layout()
    plt.savefig('hmm_sequences.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    observations, states, params = generate_dataset()
    print(f"Observations shape: {observations.shape}")  # (50, 100, 2)
    print(f"States shape: {states.shape}")              # (50, 100)
    print(f"True transition matrix:\n{params['A']}")
    visualize_sequence(observations[0], states[0])
