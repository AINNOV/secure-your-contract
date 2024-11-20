import torch
from transformers import TrainerCallback, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from omegaconf import OmegaConf
from _utils import prompt_with_template
import mlflow
import mlflow.pytorch
from transformers import TrainerCallback
import sys, os
# from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate import meteor_score
from torchmetrics.text import TranslationEditRate
from huggingface_hub import login


import nltk
nltk.download('wordnet')

class MLflowLoggingCallback(TrainerCallback):
    def __init__(self, run_name="training_run"):
        mlflow.set_tracking_uri("./mlruns")  
        mlflow.start_run(run_name=run_name)      

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):  
                    mlflow.log_metric(key, value, step=state.global_step)
    
    def on_evaluate(self, args, state, control, **kwargs):

        eval_logs = [log for log in state.log_history if "eval_loss" in log]
        if eval_logs:
            eval_loss = eval_logs[-1]["eval_loss"]  
            mlflow.log_metric("eval_loss", eval_loss, step=state.global_step)  
            print(f"🚨 Evaluation Loss: {eval_loss:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        mlflow.end_run()

# Define a custom callback for evaluation inference
class InferenceCallback(TrainerCallback):
    def __init__(self, tokenizer, model, message, GT_path, device):
        self.tokenizer = tokenizer
        self.model = model
        self.GT_path = GT_path
        self.message = message
        self.device = device

    def on_evaluate(self, args, state, control, **kwargs):
        print("\nPerforming inference after evaluation...")
        generated_texts = []
        references = []

        #for example in self.eval_dataset:

        for idx, (message, GT_path) in enumerate(zip(self.message, self.GT_path)):
            
            # contract = example["prompt"]
            # reference = example["response"]

            inputs = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=False, return_tensors="pt").to(self.device)
            generate_ids = self.model.generate(inputs)
            
            input_text = self.tokenizer.batch_decode(inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][len(input_text):]
            # print(f"🚨 Input Contract: {input_text}\n🚨 Generated Response: {output}\n")
            # print(f"🚨 Generated Response: {output}\n")


            generated_texts.append(output)
            with open(GT_path, "r") as file:
                gt = file.read()
            references.append(gt)

            with open(f"analysis_output_{state.global_step}_{idx}.txt", "w") as f:
                f.write(output)

            # Log artifacts
            # mlflow.log_artifact(f"inference_input_{idx}.txt")
            mlflow.log_artifact(f"analysis_output_{state.global_step}_{idx}.txt")
            print(f"🚨 Generated Analysis [{state.global_step}_{idx}]: {output}\n")

    
        bleu_score = corpus_bleu([[ref.split()] for ref in references], [gen.split() for gen in generated_texts])
        # print(bleu_score)
        # Calculate METEOR score
        # meteor_scores = [meteor_score.meteor_score([ref], gen) for ref, gen in zip(references, generated_texts)]
        tokenized_references = [ref.split() for ref in references]
        tokenized_generated_texts = [gen.split() for gen in generated_texts]

        # Calculate METEOR score
        meteor_scores = [
            meteor_score.meteor_score([ref], gen) 
            for ref, gen in zip(tokenized_references, tokenized_generated_texts)
]
        
        # print(meteor_scores)
        # Calculate TER score
        ter_metric = TranslationEditRate()
        ter_scores = ter_metric(generated_texts, references)
        # print(ter_scores)
        # Log the metrics to MLFlow
        mlflow.log_metric("BLEU", bleu_score, step = state.global_step)
        mlflow.log_metric("METEOR", sum(meteor_scores)/len(meteor_scores), step = state.global_step)  # Average METEOR score
        mlflow.log_metric("TER", ter_scores, step = state.global_step)
        # for idx, rouge in enumerate(rouge_scores):
        #     mlflow.log_metric(f"ROUGE_1_{idx}", rouge['rouge1'].fmeasure)
        #     mlflow.log_metric(f"ROUGE_2_{idx}", rouge['rouge2'].fmeasure)
        #     mlflow.log_metric(f"ROUGE_L_{idx}", rouge['rougeL'].fmeasure)

        # Clean up local files if needed
        for idx in range(len(self.message)):
            # os.remove(f"inference_input_{idx}.txt")
            os.remove(f"analysis_output_{state.global_step}_{idx}.txt")

# Main function to set up and start training
def main(config_path):
    # Load the configuration file
    config = OmegaConf.load(config_path)
    with open(config.hf_token, "r") as file:
        hf_token = file.read().strip()
    login(hf_token)
    
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
        eval_steps=50, 
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
        callbacks=[InferenceCallback(tokenizer, model, messages, config.eval_GT, torch.device("cuda" if torch.cuda.is_available() else "cpu")), # originally messages instead of eval_dataset["train"]
                    MLflowLoggingCallback(run_name=config.training.save_model_path)]
    )

    trainer.train()
    trainer.save_model(config.training.save_model_path)

if __name__ == "__main__":
    main("../configs/finetune.yml")