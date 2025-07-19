# 下载模型
from modelscope import snapshot_download

# https://modelscope.cn/models/google/madlad400-3b-mt
model_dir = snapshot_download('google/madlad400-3b-mt', cache_dir='./models')
print(f'模型保存在：{model_dir}')
