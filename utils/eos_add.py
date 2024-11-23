# You can use this file if your model has different end token from LLaMa-2 (</s>) #

import json

input_file = ...#
output_file = ...#

with open(input_file, "r") as f:
    data = json.load(f)

for entry in data:
    entry["response"] += " </s>"

with open(output_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"JSON file updated successfully with </s> added to each response and saved as {output_file}.")