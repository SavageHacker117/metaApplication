

import json

def load_dialogue_data(file_path):
    """
    Loads dialogue data from a JSON file.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} dialogue samples from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Check file format.")
        return []

# Example usage:
# if __name__ == "__main__":
#     # Assuming dataset_generator.py has been run to create this file
#     data = load_dialogue_data("./synthetic_dialogue_data.json")
#     if data:
#         print("First dialogue sample:")
#         print(json.dumps(data[0], indent=2))


