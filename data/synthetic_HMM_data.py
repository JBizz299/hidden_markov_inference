import numpy as np
import pandas as pd

# Observation sequence

def generate_synthetic_hmm_data(num_sequences=100, sequence_length=50, num_states=3, num_observations=5):
    
    # Initial state distribution
    initial_state_distribution = np.random.dirichlet(np.ones(num_states), size=1)[0]


    # Transition probabilities
    transition_probabilities = np.random.dirichlet(np.ones(num_states), size=num_states)

    # Emission probabilities