import os
import torch
import numpy as np
from omegaconf import OmegaConf

## model ##
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, PeftModel
 
## metric ##
from torchmetrics.text import TranslationEditRate
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score

## evaluation ##
def evaluate_finetuned_model(generated_dir, gt_dir, embedding_model_name, device):

    ## for BERT-SIM ##
    embedding_model = SentenceTransformer(embedding_model_name)

    contract_files = sorted(os.listdir(generated_dir))
    gt_files = sorted(os.listdir(gt_dir))


    assert len(contract_files) == len(gt_files), "Mismatch in number of contract and GT files"

    ## dump all the text pairs first ##
    generated_texts = []
    references = []

    for contract_file, gt_file in zip(contract_files, gt_files):
        contract_path = os.path.join(generated_dir, contract_file)
        gt_path = os.path.join(gt_dir, gt_file)

        with open(contract_path, "r") as f:
            generated_text = f.read()
        with open(gt_path, "r") as f:
            gt_text = f.read()

        ## skip if any gt is empty ##
        if gt_text.strip() == '': continue

        generated_texts.append(generated_text)
        references.append(gt_text)
    
    ## 1. BLEU (averaged on n=1 ~ 4) ##
    avg_bleu = corpus_bleu([[ref.split()] for ref in references], [gen.split() for gen in generated_texts])

    ## 2. METEOR ##
    tokenized_references = [ref.split() for ref in references]
    tokenized_generated_texts = [gen.split() for gen in generated_texts]
    meteor_scores = [
        meteor_score.meteor_score([ref], gen) 
        for ref, gen in zip(tokenized_references, tokenized_generated_texts)]
    avg_meteor = np.mean(meteor_scores)

    ## 3. TER ##
    ter_metric = TranslationEditRate()
    avg_ter = ter_metric(generated_texts, references)

    ## 4 ~ 6. ROUGE ##
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [rouge.score(gen, ref) for gen, ref in zip(generated_texts, references)]
    avg_rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    avg_rouge2 = np.mean([score['rouge2'].fmeasure for score in rouge_scores])
    avg_rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

    ## 7. BERT-SIM ##
    gt_embeddings = embedding_model.encode(references)
    gen_embeddings = embedding_model.encode(generated_texts)
    embedding_similarities = [
        cosine_similarity([gen_emb], [gt_emb])[0][0]
        for gen_emb, gt_emb in zip(gen_embeddings, gt_embeddings)
    ]
    avg_bert_similarity = np.mean(embedding_similarities)


    ## 8. Precision ##
    ## 9. Recall ##
    ## 10. F-1 ##
    y_true = [set(ref.split()) for ref in references]
    y_pred = [set(gen.split()) for gen in generated_texts]

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for true_set, pred_set in zip(y_true, y_pred):
        intersection = true_set & pred_set
        precision = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0
        recall = len(intersection) / len(true_set) if len(true_set) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    ## average across all documents ##
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)


    print("\nFinal Evaluation Scores:")
    print(f"BLEU: {avg_bleu:.4f}")
    print(f"METEOR: {avg_meteor:.4f}")
    print(f"TER: {avg_ter:.4f}")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")
    print(f"BERT-Similarity: {avg_bert_similarity:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1-Score: {avg_f1:.4f}")
   
    ## save as a file ##
    with open(f"./eval_output/{config.test_generated_path.split('/')[-1]}.txt", "w") as file:
        file.write("Final Evaluation Scores:\n")
        file.write(f"BLEU: {avg_bleu:.4f}\n")
        file.write(f"METEOR: {avg_meteor:.4f}\n")
        file.write(f"TER: {avg_ter:.4f}\n")
        file.write(f"ROUGE-1: {avg_rouge1:.4f}\n")
        file.write(f"ROUGE-2: {avg_rouge2:.4f}\n")
        file.write(f"ROUGE-L: {avg_rougeL:.4f}\n")
        file.write(f"BERT-Similarity: {avg_bert_similarity:.4f}\n")
        file.write(f"Precision: {avg_precision:.4f}\n")
        file.write(f"Recall: {avg_recall:.4f}\n")
        file.write(f"F1-Score: {avg_f1:.4f}\n")


if __name__ == "__main__":
    config_path = "./configs/evaluation_onlyscores.yml"
    config = OmegaConf.load(config_path)

    generated_dir = config.test_generated_path
    gt_dir = config.test_gt_path
    embedding_model_name = config.bert_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_finetuned_model(generated_dir, gt_dir, embedding_model_name, device)