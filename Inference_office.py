import os

from transformers import T5ForConditionalGeneration, T5Tokenizer

# https://modelscope.cn/models/google/madlad400-3b-mt
# google/madlad400-3b-mt
root_path = os.path.dirname(os.path.abspath(__file__))
model_name = os.path.join(root_path, "models/google/madlad400-3b-mt")

model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained(model_name)

text = "<2en> 腺苷酸活化蛋白激酶"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
outputs = model.generate(input_ids=input_ids)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
# ADP kinase
print(f'result: {result}')
