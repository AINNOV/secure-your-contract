# Secure Your Contract #
Secure Your Contract is an AI-based contract assistant that helps you draft a contract avoiding possible disadvantageous keywords/terms/phrases.

It is based on LLaMa2 for simplicity with much of references, especially about 4bit quantization which is very promising towards a lightweight off-the-shelf system.

**Secure Your Contract** provides two types of analysis:

1. LLM-based negative **term/phrases** detector will provide the detections against risky or weakly risky terms, the reasons for them and corresponding refinement suggestions.

2. Traditional NLP/text mining-based negative **keywords** detector consist of 10 different methodologies - please check ```./LLMS/neg_detect.py``` such that efficiently/independently provides additional helpful information in drafting your contract or agreement without any overhead such as two times of inference of LLM in total.


### Preparation ###

1. You need to follow the instruction of [MITIE](https://github.com/mit-nlp/MITIE) github such that you are able to run ```./LLMs/mitie_finetune.py```. Place ```MITIE-models``` right under the root folder.

2. You need to download 100 ```contract``` keyword-related contract/agreement documents from [wonder](https://www.wonder.legal/us) such that the model learns realistic examples. 

3. For both auto-generated (by ChatGPT in our method for **memory management**) and crawled data, contract(prompt)-analysis(response) data pairs are placed under ```./data/pair/prompt/``` and ```./data/pair/response/``` respectively with each separated as ```.txt``` files.



### Fine-Tuning ###

Fine-tune LLaMa2 7b chat model with 4 bit quantization QLoRA:

```
cd LLMs
python3 llama_finetune.py
```


### Inference ###

Inference with your fine-tuned model on the sample contract data:

```
cd LLMs
python3 finetuned_llama.py
```

Note you can change ```configs/inference.yml``` for different settings.


### Data Preparation for Fine-Tuning ###

```
cd utils
python3 save_oneliner.py
```

The code will convert the normal prompts into ones for LLaMa, saved in ```data/dump``` as a backup (not for a real usage). It also push the data to your own huggingface repository, which our code actually uses.

