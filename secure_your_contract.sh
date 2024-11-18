cd ./LLMs
python3 finetuned_llama.py --inference_config "../configs/inference.yml"
python3 neg_detect.py
