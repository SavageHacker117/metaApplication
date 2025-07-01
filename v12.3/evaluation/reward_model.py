
from transformers import pipeline

class RewardModel:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        # This is a placeholder. In a real RL-LLM, this would be a fine-tuned reward model.
        # For demonstration, we use a sentiment analysis model to simulate reward.
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

    def get_reward(self, user_query, llm_response):
        """
        Calculates a reward based on the sentiment of the LLM's response to the user query.
        Positive sentiment yields higher reward.
        """
        text_to_analyze = f"{user_query} {llm_response}"
        result = self.sentiment_pipeline(text_to_analyze)[0]
        
        if result["label"] == "POSITIVE":
            # Scale score to be between 0.5 and 1.0 for positive sentiment
            reward = 0.5 + (result["score"] * 0.5)
        else:
            # Scale score to be between 0.0 and 0.5 for negative sentiment
            reward = 0.5 - (result["score"] * 0.5)
            
        return reward

# Example usage:
# if __name__ == "__main__":
#     reward_model = RewardModel()
#     query = "I am feeling very sad today."
#     response = "I am sorry to hear that. How can I help you feel better?"
#     reward = reward_model.get_reward(query, response)
#     print(f"Reward for the response: {reward}")


