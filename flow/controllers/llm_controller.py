import math
import os
import numpy as np
from openai import OpenAI
import textwrap
import time
from pprint import pprint
from flow.controllers.base_controller import BaseController

# os.environ["http_proxy"] = "http://localhost:7890"
# os.environ["https_proxy"] = "http://localhost:7890"
# os.environ["OPENAI_API_KEY"] = ""

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
delimiter = "###########"

expname = "figure_eight_0" # the name of each exp
logdir = f"log/{expname}"
os.makedirs(logdir, exist_ok=True)

# Logging Functions
def log_scenario_description(scenario_description):
    with open(f"{logdir}/scenario_log.txt", "a") as log_file:
        log_file.write(scenario_description + "\n")

def log_agent_reasoning(veh_id, reasoning, action_id, message):
    with open(f"{logdir}/agent_{veh_id}_log.txt", "a") as log_file:
        log_file.write(f"Reasoning: {reasoning}\n")
        log_file.write(f"Action ID: {action_id}, Message: {message}\n")

def log_communication(veh_id, message):
    with open(f"{logdir}/communication_{veh_id}_log.txt", "a") as log_file:
        log_file.write(f"Message: {message}\n")

def log_decision_theory(veh_id, decision_logic):
    with open(f"{logdir}/decision_theory_{veh_id}_log.txt", "a") as log_file:
        log_file.write(f"Decision Logic: {decision_logic}\n")

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        return result
    return wrapper

def calculate_reward(env):
    reward = 0
    # Reward for maintaining safe distance and smooth traffic
    for veh_id in env.k.vehicle.get_ids():
        speed = env.k.vehicle.get_speed(veh_id)
        headway = env.k.vehicle.get_headway(veh_id)
        reward += speed - abs(headway) * 0.1

    # Penalty for collisions
    collisions = env.k.simulation.get_collision_number()
    reward -= collisions * 10
    return reward


class DriverAgent():
    def __init__(self, veh_id):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.veh_id = veh_id


# You are an autonomous driver. Please give a suitable acceleration based on the scenario description and shared message below.
        # Please output an action id and a message you want to tell other vehicles. Let's think step by step. You can give your reasoning progress.
        # Output Format: only give a dictionary as below.
        # {'action': 0, 'message': 'I suggest veh 0,1 slow down. Others please maintain the current speed.'}

    def make_decision(self, scenario_description, env):
        system_prompt = textwrap.dedent("""
            You are an autonomous driver in a traffic simulation. Your goal is to decide the best acceleration to help all vehicles pass through the intersection smoothly and safely. Based on the scenario and shared messages, provide an action and a message for other vehicles.
            Respond with a dictionary containing 'action' (0 for idle, 1 for acceleration, 2 for deceleration) and 'message' to communicate with other vehicles.
            """)

        shared_message = env.message_pool.get_msg()
        human_message = textwrap.dedent(f"""
        {delimiter} Scenario Description:
        {scenario_description}
        
        {delimiter} Shared Message:
        {shared_message}
        """)


        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": human_message,
                }
            ],
            model="gpt-3.5-turbo",
        )

        response = chat_completion.choices[0].message.content
        try:
            response = eval(response)
        except:
            print("Output Format Error: ", response)
            response = {'action': 0, 'message': ' '}
        
        action_id = response['action']
        message = response['message']
        env.message_pool.join(message, self.veh_id)

        # Loging
        log_scenario_description(scenario_description)
        log_agent_reasoning(self.veh_id, human_message, action_id, message)
        print(f"\n\n------------{self.veh_id}--------------\n\n")
        print(human_message)
        print(f"\n{delimiter}\naction_id: {action_id}\nmessage:{message}")

        return action_id

class LLMController(BaseController):
    def __init__(self, veh_id, v0=30, T=1, a=1, b=1.5, delta=4, s0=2, time_delay=0.0, noise=0, fail_safe=None, display_warnings=True, car_following_params=None):
        BaseController.__init__(self, veh_id, car_following_params, delay=time_delay, fail_safe=fail_safe, noise=noise, display_warnings=display_warnings)
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.DA = DriverAgent(veh_id)

    @measure_time
    def get_accel(self, env):
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(0, v * self.T + v * (v - lead_vel) / (2 * np.sqrt(self.a * self.b)))

        IDM_acc = self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)
        
        llm_acc = self.llm_decision(IDM_acc, env)
        return llm_acc

    def llm_decision(self, IDM_acc, env):
        state = env.get_state()
        speed, pos = state['speed'], state['pos']

        scenario_description = textwrap.dedent(f"""\
        You are driving on a road with a figure-eight pattern. There is only a single lane in each direction with an intersection.
        Your speed is {env.k.vehicle.get_speed(self.veh_id)} m/s, IDM acceleration is {IDM_acc} m/s^2, and lane position is {env.k.vehicle.get_position(self.veh_id)} m.
        Other vehicles:
        """)
        for i in range(len(speed)):
            scenario_description += f" - Vehicle {i}: speed {speed[i]} m/s, position {pos[i]} m.\n"
        
        action_space = textwrap.dedent(f"""
        {delimiter} IDM acceleration: {IDM_acc} m/s^2. Available actions:
        - IDLE (remain at current speed) - Action ID: 0
        - Accelerate - Action ID: 1
        - Decelerate - Action ID: 2
        """)
        scenario_description += action_space


        action = self.DA.make_decision(scenario_description, env)

        # Log communication and decision process
        log_communication(self.veh_id, action)
        log_decision_theory(self.veh_id, scenario_description)

        if action == 1:
            IDM_acc += abs(IDM_acc) * 0.3
        elif action == 2:
            IDM_acc -= abs(IDM_acc) * 0.3

        return IDM_acc

# Integrate reward calculation and speed measurement
def run_simulation(env, llm_controller):
    total_reward = 0
    total_speed = 0
    num_vehicles = len(env.k.vehicle.get_ids())
    for step in range(env.sim_params.simulation_step):
        env.step()
        reward = calculate_reward(env)
        total_reward += reward
        for veh_id in env.k.vehicle.get_ids():
            total_speed += env.k.vehicle.get_speed(veh_id)
        llm_controller.get_accel(env)
    average_speed = total_speed / (num_vehicles * env.sim_params.simulation_step)
    print(f"Total Reward: {total_reward}")
    print(f"Average Speed: {average_speed} m/s")
