from model import WSCNet, CAERSNet, PDANet, Stimuli_Aware_VEA

class WSCNet_Config():
    def __init__(self):
        self.model = WSCNet()
        self.epoch = 100
        self.batch_size = 128
        self.learning_rate = 1e-4

class CAERSNet_Config():
    def __init__(self):
        self.model = CAERSNet()
        self.epoch = 100
        self.batch_size = 128
        self.learning_rate = 1e-4


class PDANet_Config():
    def __init__(self):
        self.model = PDANet()
        self.epoch = 100
        self.batch_size = 128
        self.learning_rate = 1e-2    


class Stimuli_Aware_VEA_Config():
    def __init__(self):
        self.model = Stimuli_Aware_VEA()
        self.epoch = 100
        self.batch_size = 128
        self.learning_rate = 1e-2