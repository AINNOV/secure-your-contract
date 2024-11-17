import mitie

model_path = "../MITIE-models/english/total_word_feature_extractor.dat"
word_feature_extractor = mitie.total_word_feature_extractor(model_path)

trainer = mitie.text_categorizer_trainer(model_path)
trainer.add_labeled_text("The employee must work overtime without additional pay.", "negative")
trainer.add_labeled_text("Employees are entitled to annual leave as per company policy.", "neutral")
trainer.add_labeled_text("All disputes shall be resolved by the employer's discretion.", "negative")


print("Training the model...")
model = trainer.train()


model.save_to_disk("./negative_phrase_categorizer.dat")


text_categorizer = mitie.text_categorizer("negative_phrase_categorizer.dat")
test_sentence = "The employer reserves the right to amend terms at any time."
predicted_label, confidence = text_categorizer(test_sentence)

print(f"Predicted Label: {predicted_label}, Confidence: {confidence}")