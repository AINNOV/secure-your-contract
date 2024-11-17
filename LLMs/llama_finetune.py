import torch
from transformers import TrainerCallback, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from omegaconf import OmegaConf
from _utils import prompt_with_template

# Define a custom callback for evaluation inference
class InferenceCallback(TrainerCallback):
    def __init__(self, tokenizer, model, messages, device):
        self.tokenizer = tokenizer
        self.model = model
        self.messages = messages
        self.device = device

    def on_evaluate(self, args, state, control, **kwargs):
        print("\nPerforming inference after evaluation...")
        for message in self.messages:
            inputs = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=False, return_tensors="pt").to(self.device)
            generate_ids = self.model.generate(inputs)
            input_text = self.tokenizer.batch_decode(inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][len(input_text):]
            print(f"🚨 Input Contract: {input_text}\n🚨 Generated Response: {output}\n")

# Main function to set up and start training
def main(config_path):
    # Load the configuration file
    config = OmegaConf.load(config_path)
    
    # Initialize model, tokenizer, and device
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  
    model = AutoModelForCausalLM.from_pretrained(config.base_model, quantization_config=bnb_config, device_map="auto", low_cpu_mem_usage=True)

    # Prepare Lora configuration and apply it to the model
    lora_config = LoraConfig(r=config.lora.r, lora_alpha=config.lora.alpha, lora_dropout=config.lora.dropout, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_config)

    # Load datasets and preprocess
    train_dataset = load_dataset(config.dataset.train)
    eval_dataset = load_dataset(config.dataset.eval)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size_eval,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.epochs,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        fp16=config.training.fp16,
    )

    # Define the messages for evaluation callback
    messages = []
    for _path in config.eval_contract:
        with open(_path, "r") as file:
            messages.append(prompt_with_template(file.read()))

    
    # Initialize and run trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=eval_dataset["train"],
        dataset_text_field="text",
        callbacks=[InferenceCallback(tokenizer, model, messages, torch.device("cuda" if torch.cuda.is_available() else "cpu"))]
    )

    trainer.train()
    trainer.save_model(config.training.save_model_path)

if __name__ == "__main__":
    main("../configs/finetune.yml")