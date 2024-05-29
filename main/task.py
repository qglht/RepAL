from neurogym.envs.perceptualdecisionmaking import PerceptualDecisionMaking, PerceptualDecisionMakingDelayResponse
from neurogym.envs.gonogo import GoNogo
from neurogym.envs.reachingdelayresponse import ReachingDelayResponse

timing_without_delay_training = {
            'fixation': [100,200,300],
            'stimulus':  [3000,4000,5000],
            'delay': 0,
            'decision':  [100,200,300]}
timing_without_delay_testing = {
            'fixation': 200,
            'stimulus': 4000,
            'delay': 0,
            'decision': 200}
timing_with_delay_training = {
            'fixation': [100, 200,300],
            'stimulus':  [3000,4000,5000],
            'delay': [500, 1000,1500],
            'decision':  [100,200,300]}
timing_with_delay_testing = {
            'fixation': 500,
            'stimulus': 4000,
            'delay': 2000,
            'decision': 500}

class PerceptualDecisionMakingT(PerceptualDecisionMaking):
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
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
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
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
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
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
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
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
    
class GoNogoT(GoNogo):
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if self.mode == 'train':  # set a random timing for each of the periods of the tasks
            self.timing = timing_without_delay_training
        else : # set a fixed timing for each of the periods of the tasks
            self.timing = timing_without_delay_testing
        
    def _new_trial(self, **kwargs):
        trial = super()._new_trial(**kwargs)
        return trial
    
class AntiGoNogoT(GoNogo):
    """Only change groundtruth to anti-response"""
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
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
    
class GoNogoDelayResponseT(GoNogo):
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if self.mode == 'train':  # set a random timing for each of the periods of the tasks
            self.timing = timing_with_delay_training
        else : # set a fixed timing for each of the periods of the tasks
            self.timing = timing_with_delay_testing
        
    def _new_trial(self, **kwargs):
        trial = super()._new_trial(**kwargs)
        return trial
    
class AntiGoNogoDelayResponseT(GoNogo):
    """Only change groundtruth to anti-response"""
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
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
    
class ReachingDelayResponseT(ReachingDelayResponse):
    def __init__(self, config,**kwargs):
        super().__init__(dt=config["dt"], **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if self.mode == 'train':  # set a random timing for each of the periods of the tasks
            self.timing = timing_without_delay_training
        else : # set a fixed timing for each of the periods of the tasks
            self.timing = timing_without_delay_testing
        
    def _new_trial(self, **kwargs):
        trial = super()._new_trial(**kwargs)
        return trial
    
class AntiReachingDelayResponseT(ReachingDelayResponse):
    """Only change groundtruth to anti-response"""
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
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
    
class ReachingDelayResponseDelayResponseT(ReachingDelayResponse):
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if self.mode == 'train':  # set a random timing for each of the periods of the tasks
            self.timing = timing_with_delay_training
        else : # set a fixed timing for each of the periods of the tasks
            self.timing = timing_with_delay_testing
        
    def _new_trial(self, **kwargs):
        trial = super()._new_trial(**kwargs)
        return trial
    
class AntiReachingDelayResponseDelayResponseT(ReachingDelayResponse):
    """Only change groundtruth to anti-response"""
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
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
