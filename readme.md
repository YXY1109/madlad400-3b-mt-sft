# 微调翻译模型

## 执行流程

- 执行：download_model.py。下载模型
- 执行：inference_office.py。验证原始模型
- 执行：model_sft_lora.py。开始训练
- 执行：inference_office_lora.py。验证微调模型

## lora目录打包

```
tar -czvf lora-final.tar madlad-lora-zh-en-final/ 
```