
import random
import json

def generate_synthetic_dialogue_data(num_samples=100, max_turns=5):
    """
    Generates synthetic dialogue data for RL-LLM training.
    Each sample includes a user query, LLM response, and a simulated reward.
    """
    data = []
    for i in range(num_samples):
        dialogue = []
        user_query = f"User query {i+1}: How can I improve my {random.choice(['coding skills', 'writing', 'public speaking'])}?"
        llm_response = f"LLM response {i+1}: To improve your {user_query.split('my ')[1][:-2]}, you should practice regularly and seek feedback."
        
        # Simulate a reward based on response quality (e.g., length, keywords)
        reward = random.uniform(0.5, 1.0) if 'practice' in llm_response and 'feedback' in llm_response else random.uniform(0.1, 0.5)
        
        dialogue.append({"speaker": "user", "text": user_query})
        dialogue.append({"speaker": "llm", "text": llm_response})
        
        # Add more turns if max_turns > 2
        for turn in range(max_turns - 2):
            if random.random() < 0.7: # 70% chance of continuing dialogue
                follow_up_user = f"User follow-up {i+1}-{turn+1}: Can you elaborate on {random.choice(['practice techniques', 'types of feedback'])}?"
                follow_up_llm = f"LLM follow-up {i+1}-{turn+1}: Certainly, for practice, consider {random.choice(['daily exercises', 'project-based learning'])}. For feedback, peer reviews or mentor guidance are effective."
                dialogue.append({"speaker": "user", "text": follow_up_user})
                dialogue.append({"speaker": "llm", "text": follow_up_llm})
                reward += random.uniform(0.1, 0.3) # Increment reward for longer, coherent dialogues
            else:
                break

        data.append({
            "dialogue_id": f"dialogue_{i+1}",
            "dialogue": dialogue,
            "final_reward": reward
        })
    return data

if __name__ == "__main__":
    synthetic_data = generate_synthetic_dialogue_data(num_samples=500)
    output_path = "./synthetic_dialogue_data.json"
    with open(output_path, "w") as f:
        json.dump(synthetic_data, f, indent=4)
    print(f"Generated {len(synthetic_data)} samples and saved to {output_path}")


