from transformers import TrainerCallback, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  

with open("../template/template_concise.txt", "r", encoding="utf-8") as file:
    text = file.read()

tokens = tokenizer(text)["input_ids"]
num_tokens = len(tokens)

print(f"The number of tokens in the given file is: {num_tokens}")