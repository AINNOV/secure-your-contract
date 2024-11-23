import os
import json

contract_dir = '../data/test/contract'
gt_dir = '../data/test/GT'
legend_file = '../data/raw/SYC_train_with_testPDF.json'

if os.path.exists(legend_file):
    with open(legend_file, 'r', encoding='utf-8') as f:
        legend_data = json.load(f)
else:
    legend_data = []

contract_files = os.listdir(contract_dir)
gt_files = os.listdir(gt_dir)


for contract_file in contract_files:
    if contract_file.endswith('.txt'):
        contract_name = contract_file[:-4] 
        gt_file = contract_name + '.txt'

        if gt_file in gt_files:

            with open(os.path.join(contract_dir, contract_file), 'r', encoding='utf-8') as f:
                contract_content = f.read()

            with open(os.path.join(gt_dir, gt_file), 'r', encoding='utf-8') as f:
                gt_content = f.read()

            if gt_content == '': continue

            legend_data.append({
                "prompt": contract_content,
                "response":  "Hello! I am 'Secure Your Contract,' and I will help you with drafting your agreements!\n\n---\n\n" + gt_content
            })


with open(legend_file, 'w') as f:
    json.dump(legend_data, f, ensure_ascii = False, indent = 4)

print(f"{legend_file} succesfully updated")