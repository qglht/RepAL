from neurogym.envs.perceptualdecisionmaking import (
    PerceptualDecisionMaking,
    PerceptualDecisionMakingDelayResponse,
)
from neurogym.envs.gonogo import GoNogo
from neurogym.envs.reachingdelayresponse import ReachingDelayResponse
import numpy as np
import neurogym as ngym

timing_without_delay_training = {
    "fixation": [100, 200, 300],
    "stimulus": [3000, 4000, 5000],
    "delay": 0,
    "decision": [100, 200, 300],
}
timing_without_delay_testing = {
    "fixation": 200,
    "stimulus": 4000,
    "delay": 0,
    "decision": 200,
}
timing_with_delay_training = {
    "fixation": [100, 200, 300],
    "stimulus": [3000, 4000, 5000],
    "delay": [500, 1000, 1500],
    "decision": [100, 200, 300],
}
timing_with_delay_testing = {
    "fixation": 500,
    "stimulus": 4000,
    "delay": 2000,
    "decision": 500,
}


class PerceptualDecisionMakingT(PerceptualDecisionMaking):
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if (
            self.mode == "train"
        ):  # set a random timing for each of the periods of the tasks
            self.timing = timing_without_delay_training
        else:  # set a fixed timing for each of the periods of the tasks
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
        if (
            self.mode == "train"
        ):  # set a random timing for each of the periods of the tasks
            self.timing = timing_without_delay_training
        else:  # set a fixed timing for each of the periods of the tasks
            self.timing = timing_without_delay_testing

    def _new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods (in
            the case of perceptualDecisionMaking: fixation, stimulus and
            decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """
        # Trial info
        trial = {
            "ground_truth": self.rng.choice(self.choices),
            "coh": self.rng.choice(self.cohs),
        }
        trial.update(kwargs)

        coh = trial["coh"]
        ground_truth = trial["ground_truth"]
        stim_theta = self.theta[ground_truth]

        # Periods
        self.add_period(["fixation", "stimulus", "delay", "decision"])

        # Observations
        self.add_ob(1, period=["fixation", "stimulus", "delay"], where="fixation")
        stim = np.cos(self.theta - stim_theta) * (coh / 200) + 0.5
        self.add_ob(stim, "stimulus", where="stimulus")
        self.add_randn(0, self.sigma, "stimulus", where="stimulus")

        # Ground truth
        anti_ground_truth = (ground_truth + 1) % len(
            self.choices
        )  # Invert the ground truth for the anti task
        self.set_groundtruth(anti_ground_truth, period="decision", where="choice")

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period("fixation"):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards["abort"]
        elif self.in_period("decision"):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards["correct"]
                    self.performance = 1
                else:
                    reward += self.rewards["fail"]

        return self.ob_now, reward, False, {"new_trial": new_trial, "gt": gt}


class PerceptualDecisionMakingDelayResponseT(PerceptualDecisionMakingDelayResponse):
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if (
            self.mode == "train"
        ):  # set a random timing for each of the periods of the tasks
            self.timing = timing_with_delay_training
        else:  # set a fixed timing for each of the periods of the tasks
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
        if (
            self.mode == "train"
        ):  # set a random timing for each of the periods of the tasks
            self.timing = timing_with_delay_training
        else:  # set a fixed timing for each of the periods of the tasks
            self.timing = timing_with_delay_testing

    def _new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        trial = {
            "ground_truth": self.rng.choice(self.choices),
            "coh": self.rng.choice(self.cohs),
            "sigma": self.sigma,
        }
        trial.update(kwargs)

        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        periods = ["fixation", "stimulus", "delay", "decision"]
        self.add_period(periods)

        # define observations
        self.set_ob([1, 0, 0], "fixation")
        stim = self.view_ob("stimulus")
        stim[:, 0] = 1
        stim[:, 1:] = (1 - trial["coh"] / 100) / 2
        stim[:, trial["ground_truth"]] = (1 + trial["coh"] / 100) / 2
        stim[:, 1:] += self.rng.randn(stim.shape[0], 2) * trial["sigma"]

        self.set_ob([1, 0, 0], "delay")

        # Set the inverted ground truth for the decision period
        anti_ground_truth = 3 - trial["ground_truth"]
        self.set_groundtruth(anti_ground_truth, "decision")

        return trial

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False
        # rewards
        reward = 0
        # observations
        gt = self.gt_now

        if self.in_period("fixation"):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards["abort"]
        elif self.in_period("decision") and action != 0:
            new_trial = True
            if action == gt:
                reward = self.rewards["correct"]
                self.performance = 1
            elif action == 3 - gt:  # 3 - action is the other act
                reward = self.rewards["fail"]

        info = {"new_trial": new_trial, "gt": gt}
        return self.ob_now, reward, False, info


class GoNogoT(GoNogo):
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if (
            self.mode == "train"
        ):  # set a random timing for each of the periods of the tasks
            self.timing = timing_without_delay_training
        else:  # set a fixed timing for each of the periods of the tasks
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
        if (
            self.mode == "train"
        ):  # set a random timing for each of the periods of the tasks
            self.timing = timing_without_delay_training
        else:  # set a fixed timing for each of the periods of the tasks
            self.timing = timing_without_delay_testing

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            "stimulus_type": self.rng.choice(self.choices)  # 0 for Go, 1 for No-Go
        }
        trial.update(kwargs)

        # Period info
        periods = ["fixation", "stimulus", "delay", "decision"]
        self.add_period(periods)
        # set observations
        self.add_ob(1, where="fixation")
        self.add_ob(1, "stimulus", where=trial["stimulus_type"] + 1)
        self.set_ob(0, "decision")
        # if trial is GO the reward is set to R_MISS and to 0 otherwise
        self.r_tmax = self.rewards["miss"] * trial["stimulus_type"]
        self.performance = 1 - trial["stimulus_type"]

        # set ground truth during decision period
        # Invert the ground truth for the anti task
        trial["ground_truth"] = 1 - trial["stimulus_type"]
        self.set_groundtruth(trial["ground_truth"], "decision")

        return trial

    def _step(self, action):
        new_trial = False
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        if self.in_period("fixation"):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards["abort"]
        elif self.in_period("decision"):
            new_trial = True
            if action == gt:
                reward = self.rewards["correct"]
                self.performance = 1
            else:
                reward = self.rewards["fail"]
                self.performance = 0

        return ob, reward, False, {"new_trial": new_trial, "gt": gt}


class GoNogoDelayResponseT(GoNogo):
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if (
            self.mode == "train"
        ):  # set a random timing for each of the periods of the tasks
            self.timing = timing_with_delay_training
        else:  # set a fixed timing for each of the periods of the tasks
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
        if (
            self.mode == "train"
        ):  # set a random timing for each of the periods of the tasks
            self.timing = timing_with_delay_training
        else:  # set a fixed timing for each of the periods of the tasks
            self.timing = timing_with_delay_testing

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            "stimulus_type": self.rng.choice(self.choices)  # 0 for Go, 1 for No-Go
        }
        trial.update(kwargs)

        # Period info
        periods = ["fixation", "stimulus", "delay", "decision"]
        self.add_period(periods)
        # set observations
        self.add_ob(1, where="fixation")
        self.add_ob(1, "stimulus", where=trial["stimulus_type"] + 1)
        self.add_ob(0, "delay")
        self.set_ob(0, "decision")
        # if trial is GO the reward is set to R_MISS and to 0 otherwise
        self.r_tmax = self.rewards["miss"] * trial["stimulus_type"]
        self.performance = 1 - trial["stimulus_type"]

        # set ground truth during decision period
        # Invert the ground truth for the anti task
        trial["ground_truth"] = 1 - trial["stimulus_type"]
        self.set_groundtruth(trial["ground_truth"], "decision")

        return trial

    def _step(self, action):
        new_trial = False
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        if self.in_period("fixation"):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards["abort"]
        elif self.in_period("decision"):
            new_trial = True
            if action == gt:
                reward = self.rewards["correct"]
                self.performance = 1
            else:
                reward = self.rewards["fail"]
                self.performance = 0

        return ob, reward, False, {"new_trial": new_trial, "gt": gt}


class ReachingDelayResponseT(ReachingDelayResponse):
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if (
            self.mode == "train"
        ):  # set a random timing for each of the periods of the tasks
            self.timing = timing_without_delay_training
        else:  # set a fixed timing for each of the periods of the tasks
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
        if (
            self.mode == "train"
        ):  # set a random timing for each of the periods of the tasks
            self.timing = timing_without_delay_training
        else:  # set a fixed timing for each of the periods of the tasks
            self.timing = timing_without_delay_testing

    def _new_trial(self, **kwargs):
        # Trial
        trial = {"ground_truth": self.rng.uniform(self.lowbound, self.highbound)}
        trial.update(kwargs)
        ground_truth_stim = trial["ground_truth"]

        # Calculate the anti ground truth (opposite direction)
        anti_ground_truth_stim = self.highbound - (ground_truth_stim - self.lowbound)

        # Periods
        self.add_period(["stimulus", "delay", "decision"])

        self.add_ob(ground_truth_stim, "stimulus", where="stimulus")
        self.set_ob([0, -0.5], "delay")
        self.set_ob([1, -0.5], "decision")

        self.set_groundtruth([-1, -0.5], ["stimulus", "delay"])
        self.set_groundtruth([1, anti_ground_truth_stim], "decision")

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now  # 2 dim now

        if self.in_period("stimulus"):
            if not action[0] < 0:
                new_trial = self.abort
                reward = self.rewards["abort"]
        elif self.in_period("decision"):
            if action[0] > 0:
                new_trial = True
                reward = self.rewards["correct"] / ((1 + abs(action[1] - gt[1])) ** 2)
                self.performance = reward / self.rewards["correct"]

        return self.ob_now, reward, False, {"new_trial": new_trial, "gt": gt}


class ReachingDelayResponseDelayResponseT(ReachingDelayResponse):
    def __init__(self, config, **kwargs):
        super().__init__(dt=config["dt"], **kwargs)
        self.mode = config["mode"]
        self.rng = config["rng"]
        if (
            self.mode == "train"
        ):  # set a random timing for each of the periods of the tasks
            self.timing = timing_with_delay_training
        else:  # set a fixed timing for each of the periods of the tasks
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
        if (
            self.mode == "train"
        ):  # set a random timing for each of the periods of the tasks
            self.timing = timing_with_delay_training
        else:  # set a fixed timing for each of the periods of the tasks
            self.timing = timing_with_delay_testing

    def _new_trial(self, **kwargs):
        # Trial
        trial = {"ground_truth": self.rng.uniform(self.lowbound, self.highbound)}
        trial.update(kwargs)
        ground_truth_stim = trial["ground_truth"]

        # Calculate the anti ground truth (opposite direction)
        anti_ground_truth_stim = self.highbound - (ground_truth_stim - self.lowbound)

        # Periods
        self.add_period(["stimulus", "delay", "decision"])

        self.add_ob(ground_truth_stim, "stimulus", where="stimulus")
        self.set_ob([0, -0.5], "delay")
        self.set_ob([1, -0.5], "decision")

        self.set_groundtruth([-1, -0.5], ["stimulus", "delay"])
        self.set_groundtruth([1, anti_ground_truth_stim], "decision")

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now  # 2 dim now

        if self.in_period("stimulus"):
            if not action[0] < 0:
                new_trial = self.abort
                reward = self.rewards["abort"]
        elif self.in_period("decision"):
            if action[0] > 0:
                new_trial = True
                reward = self.rewards["correct"] / ((1 + abs(action[1] - gt[1])) ** 2)
                self.performance = reward / self.rewards["correct"]

        return self.ob_now, reward, False, {"new_trial": new_trial, "gt": gt}
