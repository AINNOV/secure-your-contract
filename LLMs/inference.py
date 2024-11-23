from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from omegaconf import OmegaConf
import argparse
import torch
from peft import get_peft_model, PeftModel
from _utils import prompt_with_template, set_all_seeds
import logging

logging.getLogger("nltk").setLevel(logging.WARNING)

def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    base_model = config.base_model
    input_contract_path = config.input_contract
    qlora_path = config.qlora_path

    ## model settings ##
    quantization_4bit = BitsAndBytesConfig(load_in_4bit=True)
    model_4bit = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_4bit,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    ## model load (eval mode) ##
    model_4bit = PeftModel.from_pretrained(model_4bit, qlora_path).eval()

    ## load an inference contract ##
    with open(input_contract_path, "r") as file:
        contract_text = file.read()

    ## template applied ##
    message = prompt_with_template(contract_text)
    inputs = tokenizer.apply_chat_template(
        message, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    input_len = len(
        tokenizer.batch_decode(
            inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    )

    ## inference ##
    generate_ids = model_4bit.generate(inputs.to(device))
    outputs = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0][input_len:] # cut the input prompt out

    end_event.record()
    torch.cuda.synchronize()
    inference_time = start_event.elapsed_time(end_event)

    print("\n🚨Negative Term Detector🚨 analysis:\n\n", outputs)
    # print(f"\nInference time : {(inference_time * 1e-3):.2f} sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "inference config?")
    parser.add_argument('--inference_config', type = str, default = "../configs/inference.yml")

    args = parser.parse_args()
    config = OmegaConf.load(args.inference_config)
    main(config)