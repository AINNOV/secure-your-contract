cd ./LLMs
python3 inference.py --inference_config "../configs/inference.yml"
python3 neg_detect.py --inference_config "../configs/inference.yml"
