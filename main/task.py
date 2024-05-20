from main import PerceptualDecisionMaking

class AntiPerceptualDecisionMaking(PerceptualDecisionMaking):
    """Only change groundtruth to anti-response"""
    def __init__(self, dt=100, **kwargs):
        super().__init__(dt=dt, **kwargs)
        
    def _new_trial(self, **kwargs):
        trial = super()._new_trial(**kwargs)
        trial['ground_truth'] = 1 - trial['ground_truth']
        return trial

