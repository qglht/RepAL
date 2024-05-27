from neurogym.envs.perceptualdecisionmaking import PerceptualDecisionMaking, PerceptualDecisionMakingDelayResponse

timing_without_delay_training = {
            'fixation': [100, 200,300],
            'stimulus':  [1000, 2000,3000],
            'delay': 0,
            'decision': [100, 200,300]}
timing_without_delay_testing = {
            'fixation': 100,
            'stimulus': 2000,
            'delay': 0,
            'decision': 100}
timing_with_delay_training = {
            'fixation': [100, 200,300],
            'stimulus':  [1000, 2000,3000],
            'delay': [300, 500, 700, 900, 1200, 2000, 3200, 4000],
            'decision': [100, 200,300]}
timing_with_delay_testing = {
            'fixation': 100,
            'stimulus': 2000,
            'delay': 1200,
            'decision': 100}

class PerceptualDecisionMakingT(PerceptualDecisionMaking):
    def __init__(self, config, dt=100, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if self.mode == 'train':  # set a random timing for each of the periods of the tasks
            self.timing = timing_without_delay_training
        else : # set a fixed timing for each of the periods of the tasks
            self.timing = timing_without_delay_testing
    def _new_trial(self, **kwargs):
        trial = super()._new_trial(**kwargs)
        return trial
    
class AntiPerceptualDecisionMakingT(PerceptualDecisionMaking):
    """Only change groundtruth to anti-response"""
    def __init__(self, config, dt=100, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if self.mode == 'train':  # set a random timing for each of the periods of the tasks
            self.timing = timing_without_delay_training
        else : # set a fixed timing for each of the periods of the tasks
            self.timing = timing_without_delay_testing
        
    def _new_trial(self, **kwargs):
        trial = super()._new_trial(**kwargs)
        trial['ground_truth'] = 1 - trial['ground_truth']
        return trial

class PerceptualDecisionMakingDelayResponseT(PerceptualDecisionMakingDelayResponse):
    def __init__(self, config, dt=100, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if self.mode == 'train':  # set a random timing for each of the periods of the tasks
            self.timing = timing_with_delay_training
        else : # set a fixed timing for each of the periods of the tasks
            self.timing = timing_with_delay_testing
        
    def _new_trial(self, **kwargs):
        trial = super()._new_trial(**kwargs)
        return trial
    
class AntiPerceptualDecisionMakingDelayResponseT(PerceptualDecisionMakingDelayResponse):
    """Only change groundtruth to anti-response"""
    def __init__(self, config, dt=100, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if self.mode == 'train':  # set a random timing for each of the periods of the tasks
            self.timing = timing_with_delay_training
        else : # set a fixed timing for each of the periods of the tasks
            self.timing = timing_with_delay_testing
        
    def _new_trial(self, **kwargs):
        trial = super()._new_trial(**kwargs)
        trial['ground_truth'] = 1 - trial['ground_truth']
        return trial