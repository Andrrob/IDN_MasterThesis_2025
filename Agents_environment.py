import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

from Agent import discretize_state, state_bins, state_history

#from test import OpenCallProtestsApp


# This script defines the environment of the Q-learning agent
# Written by Andrea Robinson 

class WarsawProtestEnv(gym.Env):
    def __init__(self, lock, script_test_instance):
        super(WarsawProtestEnv, self).__init__()

        self.lock = lock 

        # Observation space: HRV (0 to 9) and distance (0 to 9)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([4, 10]), dtype=np.float32)
        
        # Four discrete actions 
        self.action_space = spaces.Discrete(4)
        self.actions_mapping = {
            0: ("character_walk_to_object", "neil_agent self 1.3 Walk"), # The agent´s avarar walks to participant
            1: ("character_walk_to_coordinates", "neil_agent 180,1.69,58 1.3 Walk"), # The agent's avatar walks to a point near the midpoint between the center of the protest and the participant's starting position.
            2: ("look_at_character", "neil_agent self EyesAndHead"), # The agent's avatar look at the participant. 
            3: ("play_recording", "neil_agent WarsawTestRecording_0001 0.1 false"), # The agent's avatar makes a motion resembling a beckoning gesture, intended to encourage others to follow.
        }
           
        self.state = None
        self.timestep = 0
        self.script_test_instance = script_test_instance # Allows the WarsawProtestEnv to access and modify the "is_action_complet" flag defined in test.py

    def get_real_time_data(self, vec_hrv, vec_distances):
        time.sleep(5)
        with self.lock:
            if len(vec_distances) > 0 and len(vec_hrv) > 0:  # Check if there are any observations of distance and puls available
                #print(f"New state fetched")
                self.state = np.array([vec_hrv[-1], vec_distances[-1]], dtype=np.float32)
                self.state = discretize_state(self.state, state_bins)
                state_history.append(self.state)
                return self.state
                #return vec_hrv[-1], vec_distances[-1] # The last observation is the new state
            else:
                raise ValueError("No distances available for retrieval.")
    

    def step(self, action, actuators, vec_hrv, vec_distances, step):
        # Check if action is valid 
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Execute the action 
        command, params = self.actions_mapping[action] 
        actuators.PushCommand(command, params) # Push action
        event_string = str(action) + str(step)  
        actuators.PushCommand("send_event", event_string) # Event handler sets "is_sction_complet" flag to 1 when the action has been completed. 
       
        print(f"Executing action {step}, {action}: {command}, {params}")
        #while not self.script_test_instance.is_action_completed: # Wait until event handler has set "is_sction_complet" flag to 1
            #print("Waiting for action to complete...")
        print("Action completed, continuing with the next step.")
        time.sleep(3)
        self.script_test_instance.is_action_completed = False #set the if comlpeted flag back to false
        time.sleep(2)
        
        # Get participant's response (distance and puls) to the action and update the current state
        
        #hrv, distance = self.get_real_time_data(vec_hrv, vec_distances)
        #self.state = np.array([hrv, distance], dtype=np.float32)
        #state = discretize_state(state, state_bins)

        self.state = self.get_real_time_data(vec_hrv, vec_distances)
        _ , distance = self.state 
        
        # Define the reward based on the human response
        if distance <= 1:  # If the agent is very close, give high reward
            reward = 15
        elif distance == 2:
            reward = 7
        elif distance == 3:
            reward = 5
        elif distance == 4:
            reward = 0
        elif distance >= 5:
            reward = - 10


        # Return the new state and reward 
        return self.state, reward, False, {}
