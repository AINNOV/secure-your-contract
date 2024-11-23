import mitie
import os

## follow the intruction of the official MITIE repository ##
model_path = "../MITIE-models/english/total_word_feature_extractor.dat"
word_feature_extractor = mitie.total_word_feature_extractor(model_path)
trainer = mitie.text_categorizer_trainer(model_path)

## load dataset ##
prompt_dir = "../data/pair/prompt"
label_dir = "../data/pair/labels" # negative / neutral binary tags in line-level


for filename in os.listdir(prompt_dir):
    if filename.endswith(".txt"):
        prompt_file_path = os.path.join(prompt_dir, filename)
        label_file_path = os.path.join(label_dir, filename)
        
        ## load each contract / label ##
        with open(prompt_file_path, "r", encoding="utf-8") as prompt_file:
            contract_text = prompt_file.readlines()
        with open(label_file_path, "r", encoding="utf-8") as label_file:
            labels = label_file.readlines()
        
        ## add contract term - label pairs to MITIE ##        
        for sentence, label in zip(contract_text, labels):
            label = label.strip() 
            sentence = sentence.strip()  
            trainer.add_labeled_text(sentence, label)

## MITIE train ##
print("Training the model")
model = trainer.train()
model.save_to_disk("./finetuned_mitie.dat")

## inference test ##
text_categorizer = mitie.text_categorizer("./finetuned_mitie.dat")
test_sentence = "The employer reserves the right to amend terms at any time."
predicted_label, confidence = text_categorizer(test_sentence)

print(f"Predicted Label: {predicted_label}, Confidence: {confidence}")