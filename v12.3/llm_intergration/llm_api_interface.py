

import os
from openai import OpenAI

class LLMAPIInterface:
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        self.api_key = api_key if api_key else os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables.")
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def generate_response(self, prompt, temperature=0.7, max_tokens=150):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response from LLM: {e}")
            return ""

# Example usage:
# if __name__ == "__main__":
#     # Make sure to set OPENAI_API_KEY environment variable or pass it directly
#     llm_interface = LLMAPIInterface(model_name="gpt-3.5-turbo")
#     response = llm_interface.generate_response("What is the capital of France?")
#     print(f"LLM Response: {response}")


