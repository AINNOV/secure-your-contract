import sys, os
import numpy as np
from omegaconf import OmegaConf

## LLM ##
from huggingface_hub import login
import torch
from transformers import TrainerCallback, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from _utils import prompt_with_template

## logging ## 
import mlflow
import mlflow.pytorch
from transformers import TrainerCallback

## evaluation metric ##
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.text import TranslationEditRate

import nltk
nltk.download('wordnet')

## loss logging ##
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
        pass

    def on_train_end(self, args, state, control, **kwargs):
        mlflow.end_run()

## metric / genertated analyses tracking ##
class InferenceCallback(TrainerCallback):
    def __init__(self, tokenizer, model, bert_sim, message, GT_path, device):
        self.tokenizer = tokenizer
        self.model = model
        self.GT_path = GT_path
        self.message = message
        self.device = device
        self.embedding_model = SentenceTransformer(bert_sim)

    def on_evaluate(self, args, state, control, **kwargs):
        eval_logs = [log for log in state.log_history if "eval_loss" in log]
        if eval_logs:
            eval_loss = eval_logs[-1]["eval_loss"]  
            mlflow.log_metric("eval_loss", eval_loss, step=state.global_step)  
            print(f"🚨 Evaluation Loss: {eval_loss:.4f}")
            
        print("\nPerforming inference after evaluation...")

        generated_texts = []
        references = []

        for idx, (message, GT_path) in enumerate(zip(self.message, self.GT_path)):
            ## inference ##
            inputs = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=False, return_tensors="pt").to(self.device)
            generate_ids = self.model.generate(inputs)
            input_text = self.tokenizer.batch_decode(inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][len(input_text):]

            ## stack to a container list ##
            generated_texts.append(output)
            with open(GT_path, "r") as file:
                gt = file.read()
            references.append(gt)

            with open(f"analysis_output_{state.global_step}_{idx}.txt", "w") as f: # removed later
                f.write(output)

            ## record a generated analysis as an artifact ##
            mlflow.log_artifact(f"analysis_output_{state.global_step}_{idx}.txt")
            print(f"🚨 Generated Analysis [{state.global_step}_{idx}]: {output}\n")


        ## 1. BLEU (averaged on n = 1 ~ 4) 
        bleu_score = corpus_bleu([[ref.split()] for ref in references], [gen.split() for gen in generated_texts])

        ## 2. METEOR 
        tokenized_references = [ref.split() for ref in references]
        tokenized_generated_texts = [gen.split() for gen in generated_texts]
        meteor_scores = [
            meteor_score.meteor_score([ref], gen) 
            for ref, gen in zip(tokenized_references, tokenized_generated_texts)
        ]
        
        ## 3. TER
        ter_metric = TranslationEditRate()
        ter_scores = ter_metric(generated_texts, references)

        ## 4 ~ 6. ROUGE
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [rouge.score(gen, ref) for gen, ref in zip(generated_texts, references)]
        avg_rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
        avg_rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
        avg_rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

        ## 7. Bert-Emb Sim
        gt_embeddings = self.embedding_model.encode(references)
        gen_embeddings = self.embedding_model.encode(generated_texts)
        embedding_similarities = [
            cosine_similarity([gen_emb], [gt_emb])[0][0]
            for gen_emb, gt_emb in zip(gen_embeddings, gt_embeddings)
        ]
        avg_embedding_similarity = np.mean(embedding_similarities)

        ## log to mlflow ##
        mlflow.log_metric("BLEU", bleu_score, step = state.global_step)
        mlflow.log_metric("METEOR", sum(meteor_scores)/len(meteor_scores), step = state.global_step)  # Average METEOR score
        mlflow.log_metric("TER", ter_scores, step = state.global_step)
        mlflow.log_metric("ROUGE-1", avg_rouge1, step=state.global_step)
        mlflow.log_metric("ROUGE-2", avg_rouge2, step=state.global_step)
        mlflow.log_metric("ROUGE-L", avg_rougeL, step=state.global_step)
        mlflow.log_metric("BERT-Sim", avg_embedding_similarity, step=state.global_step)

        ## delete artifacts from local after logging ##
        for idx in range(len(self.message)):
            os.remove(f"analysis_output_{state.global_step}_{idx}.txt")


def main(config_path):

    config = OmegaConf.load(config_path)
    with open(config.hf_token, "r") as file:
        hf_token = file.read().strip()
    login(hf_token)
    
    ## model / token initialization ##
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  
    model = AutoModelForCausalLM.from_pretrained(config.base_model, quantization_config = bnb_config, device_map = "auto", low_cpu_mem_usage = True)

    ## lora config applied to model ##
    lora_config = LoraConfig(r=config.lora.r, lora_alpha=config.lora.alpha, lora_dropout=config.lora.dropout, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_config)

    ## load train / eval dataset ##
    train_dataset = load_dataset(config.dataset.train)
    eval_dataset = load_dataset(config.dataset.eval)

    ## training arguments ##
    training_args = TrainingArguments(
        output_dir = config.training.output_dir,
        evaluation_strategy = "steps",
        eval_steps = config.training.logging_steps, # log and eval at the same step
        per_device_train_batch_size = config.training.batch_size,
        per_device_eval_batch_size = config.training.batch_size_eval,
        learning_rate = config.training.learning_rate,
        num_train_epochs = config.training.epochs,
        logging_steps = config.training.logging_steps,  # log and eval at the same step
        save_steps = config.training.save_steps,
        fp16 = config.training.fp16,
    )

    ## eval contracts for eval step ##
    messages = []
    for _path in config.eval_contract:
        with open(_path, "r") as file:
            messages.append(prompt_with_template(file.read())) # different custom template for different data

    ## Supervised Fine Tuning Trainer load ##
    trainer = SFTTrainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset["train"],
        eval_dataset = eval_dataset["train"],
        dataset_text_field = "text",
        callbacks = [InferenceCallback(tokenizer, model, config.bert_sim, messages, config.eval_GT, torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                    MLflowLoggingCallback(run_name=config.training.save_model_path)]
    )

    trainer.train()
    trainer.save_model(config.training.save_model_path)


if __name__ == "__main__":
    main("../configs/finetune.yml")