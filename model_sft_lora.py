import os

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

"""
中文转英文，中药测试集
https://modelscope.cn/datasets/iic/WMT-Chinese-to-English-Machine-Translation-Medical
"""

root_path = os.path.dirname(os.path.abspath(__file__))
# 完全模型保存路径
model_save_path = os.path.join(root_path, "models/madlad-lora-zh-en-final")
# Lora模型保存路径
lora_save_path = os.path.join(root_path, "models/madlad-lora-zh-en")

# 训练数据
root_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(root_path, "train_csv/damo_mt_testsets_zh2en_medical_wmt19.csv")

# 中文转英文的数据集
dataset = load_dataset('csv', data_files={'train': csv_path})['train']
dataset = dataset.train_test_split(test_size=0.1, seed=42)

root_path = os.path.dirname(os.path.abspath(__file__))
model_name = os.path.join(root_path, "models/google/madlad400-3b-mt")

tokenizer = AutoTokenizer.from_pretrained(model_name)
print("1，原始模型和分词器，加载完成")


def preprocess_function(examples):
    inputs = [f"<2en> {pair}" for pair in examples["zh"]]
    targets = [y for y in examples["en"]]

    model_inputs = tokenizer(
        inputs,
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        targets,
        max_length=256,
        truncation=True,
        padding="max_length"
    )
    labels = labels["input_ids"]
    labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
    model_inputs["labels"] = labels
    return model_inputs


tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True
)
print("2，数据处理完成")

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,  # LoRA矩阵的秩
    lora_alpha=16,  # 缩放因子
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM  # 指定任务类型
)
print("3，LoRA配置完成")

# 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数比例
print("4，模型加载完成")

training_args = Seq2SeqTrainingArguments(
    output_dir=model_save_path,  # 模型保存路径
    learning_rate=1e-5,  # 学习率
    per_device_train_batch_size=2,  # 根据GPU显存调整
    per_device_eval_batch_size=2,
    weight_decay=0.01,  # 权重衰减
    save_total_limit=3,  # 最大保存检查点数
    num_train_epochs=3,  # 训练轮数
    predict_with_generate=True,  # 生成式预测
    fp16=False,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    gradient_accumulation_steps=4,  # 梯度累积解决显存限制
    report_to="swanlab",
    logging_strategy="steps",
    logging_steps=100,  # 日志记录间隔
    logging_dir="./logs"
)
print("5，训练参数设置完成")

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True
)
print("6，数据整理器创建完成")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer
)
print("7，训练器创建完成")

print("开始训练...")
trainer.train()
print("8，训练完成")

model.save_pretrained(lora_save_path)
tokenizer.save_pretrained(lora_save_path)
print("9，Lora模型保存成功！")
