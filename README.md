[**中文**](https://github.com/fengyh3/llama_inference/blob/main/README.md) | [**English**](https://github.com/fengyh3/llama_inference/blob/main/README_en.md) 

## 基于TencentPretrain的LLaMa推理 

本项目主要支持基于[TencentPretrain](https://github.com/Tencent/TencentPretrain)的LLaMa模型量化推理以及简单的微服务部署。也可以扩展至其他模型，持续更新中。 

<br>

### 特性 
- __Int8推理__ 支持bitsandbytes库的int8推理，相比tencentpretrain中的LM推理脚本，加入了Batch推理。 
- __优化推理逻辑__ 在Multi-head Attention中加入了key和value的cache，每次inference只需要输入新生成的token。 
- __大模型多卡推理__ 支持张量并行的多卡推理。
- __微服务部署__ 支持简单的flask部署。
- __LoRA模型推理__ 施工中，计划支持使用LoRA训练的模型。 

tips：当前脚本只支持cuda推理，未来计划更多的量化部署推理的功能，敬请期待。 

<br>

### 依赖环境 
* Python >= 3.7
* torch >= 1.9
* bitsandbytes
* argparse

<br>

### 输入参数参考
* __--load_model_path__ （必填项），预训练好的模型，默认是fp16的（如果需要fp32，修改llama_infer.py的L41为对应的精度）
* __--test_path__ （必填项），输入的prompts，每一行是一个prompts。
* __--prediction_path__ （必填项），输出结果保存的路径。
* __--config_path__ （必填项），模型参数配置文件，可以保存在config文件夹中。
* __--spm_model_path__ （必填项），模型tokenizer存放的路径。
* __--batch_size__ （可选），默认为1。批处理大小，注意按需使用，因为attention cache会根据这个大小来构造tensor并且保存在显存中。
* __--seq_length__ （可选），默认为128。生成句子的总长度，等于prompts + 模型生成的长度。
* __--use_int8__ （可选），默认为False。是否使用int8推理。
* __--top_k__ （可选），默认为40。句子的生成会针对top_k做采样，影响生成多样性。
* __--top_p__ （可选），默认为0.95。句子的生成会针对累积概率top_p做采样，影响生成多样性。
* __--temperature__ （可选），默认为0.8。对最后的probabilities做一次放缩，影响token采样结果。
* __--repetition_penalty_range__ （可选），默认为1024。重复出现token的惩罚范围。
* __--repetition_penalty_slope__ （可选），默认为0。重复出现token的惩罚slope。
* __--repetition_penalty__ （可选），默认为1.15。重复出现token的惩罚系数。

<br>

### 快速开始 
#### FP16/Int8推理 
fp16推理：
```commandline
python llama_infer.py --test_path ./prompts.txt --prediction_path ./result.txt  \
                      --load_model_path xxx.bin \
                      --config_path ./config/llama_7b_config.json \
                      --spm_model_path ./tokenizer.model
``` 

如果要使用int8推理的话，加入--use_int8: 
```commandline
python llama_infer.py --test_path ./prompts.txt --prediction_path ./result.txt  \
                      --load_model_path xxx.bin --use_int8 \
                      --config_path ./config/llama_7b_config.json \
                      --spm_model_path ./tokenizer.model
```

<br>

#### 微服务部署 
需要安装flask
```commandline
pip install flask
python llama_server.py --load_model_path xxxx.bin \
                       --config_path config.json \
                       --spm_model_path tokenizer.model
```
查询命令：
```commandline
curl -H 'Content-Type: application/json' http://127.0.0.1:8888/chat -d '{"question": "xxx"}' 
```

<br>

#### 多卡张量并行推理
需要安装tensor_parallel
参数world_size为希望使用多少gpu（gpu的id从0开始）
```commandline
pip install tensor_parallel
python llama_infer_tp.py --test_path ./prompts.txt --prediction_path ./result.txt \
                         --load_model_path xxxx.bin \
                         --config_path config.json \
                         --spm_model_path tokenizer.model \
                         --world_size 2
```

