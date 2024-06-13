import math
import os
import numpy as np
from openai import OpenAI
import textwrap
from pprint import pprint

from flow.controllers.base_controller import BaseController


os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
os.environ["OPENAI_API_KEY"] = 'sk-proj-9Z4RykhKhT5eS26AGh1aT3BlbkFJ4ITIL3CAAp8beQkd6AVv'

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

delimiter = "###########"
class DriverAgent():
    def __init__(self, veh_id):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.veh_id = veh_id
    
    def make_decision(self, scenario_description, env):
        system_prompt = textwrap.dedent("""
        You are a autonomous driver. please give a suitable accelaration based on the scenario description and shared message below.
        Please output an action id and a message you wanna tell other vehicles. Let's think step by step. You can give your reasoning progress.
        Output Format: only give a dictionary as below.
        {'action': 0, 'message': 'I suggest veh 0,1 slow down. Others please maintain the current speed.'}
        """)
        shared_message = env.message_pool.get_msg()
        human_message = textwrap.dedent(f"""
        {system_prompt}\n
        {delimiter} scenario_description:\n{scenario_description}\n
        {delimiter} shared message:\n{shared_message}\n
        """)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": human_message,
                }
            ],
            model="gpt-3.5-turbo",
        )

        # TODO: separate action and shared message by a structure output
        response = chat_completion.choices[0].message.content
        try:
            response = eval(response)
        except:
            print("Output Format Error: ", response)
            response = {'action': 0, 'message': ' '}
        
        action_id = response['action']
        message = response['message']
        env.message_pool.join(message, self.veh_id)

        print(f"\n\n------------{self.veh_id}--------------\n\n")
        print(human_message)
        print(f"\n{delimiter}\naction_id: {action_id}\nmessage:{message}")

        return action_id



class LLMController(BaseController):
    """Intelligent Driver Model (IDM) controller.

    For more information on this controller, see:
    Treiber, Martin, Ansgar Hennecke, and Dirk Helbing. "Congested traffic
    states in empirical observations and microscopic simulations." Physical
    review E 62.2 (2000): 1805.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.param.SumoCarFollowingParams
        see parent class
    v0 : float
        desirable velocity, in m/s (default: 30)
    T : float
        safe time headway, in s (default: 1)
    a : float
        max acceleration, in m/s2 (default: 1)
    b : float
        comfortable deceleration, in m/s2 (default: 1.5)
    delta : float
        acceleration exponent (default: 4)
    s0 : float
        linear jam distance, in m (default: 2)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True,
                 car_following_params=None):
        """Instantiate an IDM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0

        self.DA = DriverAgent(veh_id)

    def get_accel(self, env):
        """See parent class."""
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        IDM_acc =  self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)

        llm_acc = self.llm_decision(IDM_acc, env)
        return llm_acc

    def llm_decision(self, IDM_acc, env):
        """LLM help IDM controller to make decision"""
        state = env.get_state()
        speed, pos = state['speed'], state['pos']

        scenario_description = textwrap.dedent(f"""\
        You are driving on a road like figure eight. There is only a single lane in one direction with an intersection.
        Your target is to decide your acceleration to help all vehicles pass the intersection quickly and smoothly.
        Your speed is {env.k.vehicle.get_speed(self.veh_id)} m/s, IDM acceleration is {IDM_acc} m/s^2, and lane position is {env.k.vehicle.get_position(self.veh_id)} m. 
        There are other vehicles driving around you, and below is their basic information:
        """)
        for i in range(len(speed)):
            scenario_description += f" - Vehicle `{i}` is driving on the same lane of you. The speed of it is {speed[i]} m/s, and lane position is {pos[i]} m.\n"
        
        action_space = textwrap.dedent(f"""
        {delimiter} IDM gives acceleration {IDM_acc} m/s^2. Your available actions:
        IDLE - remain in the current lane with current speed Action_id: 0
        Acceleration - accelerate the vehicle Action_id: 1
        Deceleration - decelerate the vehicle Action_id: 2
        """)
        scenario_description += action_space

        action = self.DA.make_decision(scenario_description, env)
        if action == 1:
            IDM_acc += abs(IDM_acc) * 0.3
        elif action == 2:
            IDM_acc -= abs(IDM_acc) * 0.3

        return IDM_acc


