
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch

class LLMFineTuner:
    def __init__(self, model_name="gpt2", output_dir="./llm_finetuned_models"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.output_dir = output_dir

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def prepare_dataset(self, dialogues):
        texts = []
        for dialogue in dialogues:
            formatted_dialogue = ""
            for turn in dialogue["dialogue"]:
                formatted_dialogue += f"{turn["speaker"]}: {turn["text"]}\n"
            texts.append(formatted_dialogue)

        # Tokenize texts
        tokenized_inputs = self.tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        
        # Create a Hugging Face Dataset
        dataset = Dataset.from_dict({
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_inputs["input_ids"] # For causal LM, labels are usually input_ids
        })
        return dataset

    def fine_tune(self, train_dataset, num_train_epochs=3, per_device_train_batch_size=4):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=500,
            report_to="none" # Disable integrations like Weights & Biases
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )

        print("Starting LLM fine-tuning...")
        trainer.train()
        print("LLM fine-tuning finished.")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Fine-tuned LLM saved to {self.output_dir}")

# Example usage:
# if __name__ == "__main__":
#     from data.dataset_generator import generate_synthetic_dialogue_data
#     from data.data_loader import load_dialogue_data
#     import os

#     # Generate some synthetic data if not already present
#     if not os.path.exists("./synthetic_dialogue_data.json"):
#         synthetic_data = generate_synthetic_dialogue_data(num_samples=10)
#         with open("./synthetic_dialogue_data.json", "w") as f:
#             json.dump(synthetic_data, f, indent=4)

#     dialogues = load_dialogue_data("./synthetic_dialogue_data.json")

#     if dialogues:
#         finetuner = LLMFineTuner(model_name="gpt2", output_dir="./finetuned_gpt2")
#         train_dataset = finetuner.prepare_dataset(dialogues)
#         finetuner.fine_tune(train_dataset, num_train_epochs=1)
#     else:
#         print("No dialogue data to fine-tune on.")


