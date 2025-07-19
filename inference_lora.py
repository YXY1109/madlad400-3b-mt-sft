import os

from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

# 原始模型路径
root_path = os.path.dirname(os.path.abspath(__file__))
model_name = os.path.join(root_path, "models/google/madlad400-3b-mt")
lora_name = os.path.join(root_path, "models/madlad-lora-zh-en-final")

# 加载基础模型和分词器
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(model, lora_name)

# 准备输入文本
text = "<2en> 腺苷酸活化蛋白激酶"
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

# 生成输出
outputs = model.generate(
    input_ids=input_ids,
    max_length=50,
    num_beams=4,
    early_stopping=True
)

# 解码输出
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'result: {result}')
