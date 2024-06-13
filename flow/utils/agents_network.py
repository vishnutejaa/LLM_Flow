'''
We define the method of comunication of agents here.
'''

class message_pool:
    def __init__(self):
        self.msg = {}

    def get_msg(self):
        return self.msg

    def join(self, message, veh_id):
        self.msg[veh_id] = message

