

class TextEnvironment:
    def __init__(self, initial_prompt="Hello, how can I help you?"):
        self.history = [{"role": "system", "content": initial_prompt}]
        self.current_state = initial_prompt
        self.turn = 0

    def reset(self):
        self.history = [{"role": "system", "content": "Hello, how can I help you?"}]
        self.current_state = "Hello, how can I help you?"
        self.turn = 0
        return self.current_state

    def step(self, action, user_input=None):
        # action here is the LLM's response
        self.history.append({"role": "assistant", "content": action})
        self.turn += 1

        # Simulate user input or a predefined next state
        if user_input:
            self.history.append({"role": "user", "content": user_input})
            self.current_state = user_input
        else:
            # For simplicity, let's assume a fixed user response or end of interaction
            if self.turn < 3:
                self.current_state = "Tell me more about that."
                self.history.append({"role": "user", "content": self.current_state})
            else:
                self.current_state = "<END_OF_DIALOGUE>"

        # Reward calculation (can be replaced by a more sophisticated reward model)
        reward = 1.0 if "help" in action.lower() else 0.1
        done = self.current_state == "<END_OF_DIALOGUE>"

        return self.current_state, reward, done, {}

    def get_state(self):
        return self.current_state

    def get_history(self):
        return self.history


