import pickle
import numpy as np
import os
import time

# This script defines the agent based on the Q-learning algorithm
# The enironment of the agent is defined in Agents_environemnt.py
# Written by Andrea Robinson 

# Q-learning parameters
alpha = 0.4  # Learning rate
gamma = 0.7  # Discount factor
epsilon = 0.0  # Exploration rate

# Number of bins for discretizing the state (hrv, distance) to a deiscret state space
state_bins = [4, 6]  # hrv has 4 bins, distance has 6 bins
actions = 4  # Number of possible actions
state_history = []
current_action = []

# Q-table file path
q_table_file =  # Q-table pickle file here

# Initialize the Q-table 
def initialize_q_table():
    if os.path.exists(q_table_file):
        # Load Q-table from file if it exists
        with open(q_table_file, 'rb') as f:
            return pickle.load(f)
    else:
        # Create a new Q-table if it doesn't exist
        return np.zeros(state_bins + [actions])


def save_q_table(Q_table):
    with open(q_table_file, 'wb') as f:
        pickle.dump(Q_table, f)


# Function to normalize HRV values to a scale from 0 to 3
def normalize_hrv(hrv):
    # Defining stress levels based on the baseline lnRMSSD = 3.72 # The Pooled average form this study doi: 10.1016/j.autneu.2008.10.011
    if hrv < 3.00:  # More than 20% below baseline (High Stress)
        return 3  
    elif 3.00 <= hrv < 3.35:  # 10-20% below baseline (Moderate Stress)
        return 2  
    elif 3.35 <= hrv < 4.09:  # ±10% of baseline (Normal Stress)
        return 1  
    else:  # More than 10% above baseline (Low Stress / Recovery)
        return 0   


    
def normalize_distance(distance):
    if distance <= 18:
        return 0
    elif distance >= 23:
        return 5
    else:
        # Linear mapping between 18 (0) and 23 (5)
        return round(((distance - 18) / (23 - 18)) * 5)

# Discretize the hrv and distance into bins
def discretize_state(state, bins):
    HRV_value, distance = state
    
    # Pulse: Normalize between 20 bpm and 250 bpm
    HRV_state = normalize_hrv(HRV_value) 

    distance_state = normalize_distance(distance)
    print(f"Distance: {distance_state}, HRV: {HRV_state}")
    
    return (HRV_state, distance_state)
   

# Epsilon-greedy action selection
def choose_action(state, epsilon, Q_table):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, actions)  # Choose random action
    else:
        return np.argmax(Q_table[state])  # Choose the action with highest Q-score
    
# Update the Q-table based on the agent's reward 
def update_Q_table(state, action, reward, next_state, Q_table):
    best_next_action = np.argmax(Q_table[next_state])
    Q_table[state][action] += alpha * (reward + gamma * Q_table[next_state][best_next_action] - Q_table[state][action])

# Q-learning training function (per participant session)
def train_agent(env, actuators, vec_hrv, vec_distances, training_finish): # vec_puls and vec_distances contains the observed respons of the participant, both defined in test.py 
    # Load or initialize the Q-table
    Q_table = initialize_q_table()

    
    # Get the state (participant´s hrv and distance to protest)
    state = env.get_real_time_data(vec_hrv, vec_distances)
    #state = np.array([hrv, distance], dtype=np.float32)
    #state = discretize_state(state, state_bins)

    # Perform 9 actions 
    for step in range(9):

        action = choose_action(state, epsilon, Q_table)
        current_action.append(action)
        print(f"Current action: {current_action[-1]}")
        # Perform the action in the environment and observe the human's response
        next_state, reward, _, _ = env.step(action, actuators, vec_hrv, vec_distances, step)
        if action == 2:
            time.sleep(5)
            actuators.PushCommand("free_look_at", "neil_agent") 
            event_string = "free_look" + str(step) 
            actuators.PushCommand("send_event", event_string)
        
        #next_state = discretize_state(next_state, state_bins)
        #current_action = None
        # Update the Q-table based on the reward and new state
        update_Q_table(state, action, reward, next_state, Q_table)

        # Move to the next state
        state = next_state
        print(f"Step number: {step}")

    # After completing the episode (9 actions), save the Q-table
    save_q_table(Q_table)

    print("Training completed for this episode (9 actions). Q-table saved.")
    training_finish = True
