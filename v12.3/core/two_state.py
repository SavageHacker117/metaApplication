import copy
import json

class TwoStateWorld:
    """
    Manages a production (green) and sandbox (blue) world state.
    All changes and new agent/AI/user code must pass through blue,
    and only get promoted to green if validation passes.
    """

    def __init__(self, initial_state):
        # The "production" world
        self.green = copy.deepcopy(initial_state)
        # The "sandbox"/test world
        self.blue = copy.deepcopy(initial_state)
        # History of blue state changes (for rollback/debug)
        self.history = []

    def propose_change(self, change_fn, desc=""):
        """
        Propose a change to blue state.
        change_fn: a function that modifies the blue state in-place.
        desc: description of change for audit/history.
        """
        state_before = copy.deepcopy(self.blue)
        change_fn(self.blue)
        self.history.append({
            "desc": desc,
            "change_fn": change_fn,
            "before": state_before,
            "after": copy.deepcopy(self.blue)
        })

    def validate_blue(self, validators=[]):
        """
        Run all validators on blue state.
        Returns True if all pass, False otherwise.
        """
        for validate in validators:
            if not validate(self.blue):
                return False
        return True

    def promote_blue_to_green(self):
        """Promote blue state to green (if validated)."""
        self.green = copy.deepcopy(self.blue)

    def rollback_blue(self, idx=-1):
        """
        Rollback blue state to a previous checkpoint in history.
        idx: which history entry to roll back to (default is last)
        """
        if self.history:
            self.blue = copy.deepcopy(self.history[idx]["before"])

    def diff_states(self):
        """Return a JSON diff (very simple)."""
        return json.dumps({
            "green": self.green,
            "blue": self.blue
        }, indent=2)
