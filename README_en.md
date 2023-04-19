## LLaMa Inference For TencentPretrain 

This project mainly supports LLaMa Inference and Microservice deployment based on [TencentPretrain](https://github.com/Tencent/TencentPretrain).

<br>

### Feature 
- __Int8 Inference__ Supports int8 inference with the bitsandbytes library, and adds batch inference compared to the LM inference script in tencentpretrain.  
- __Optimized Inference__ Added cache for key and value in Multi-head Attention, requiring only the newly generated token to be input for each inference. 
- __LLM Multi-Gpu Inference__ Supports tensor parallel multi-gpu inference.
- __Microservices__ To be continued. 
- __LoRA model Inference__ To be continued. 

tips: need cuda. 

<br> 

### Requirements 
* Python >= 3.7 
* torch >= 1.9 
* bitsandbytes 
* argparse 

<br>

### Input Parameters 
* __--load_model_path__ (Required) pretrained model, default by fp16. 
* __--test_path__ (Required) input prompts，one prompt each line. 
* __--prediction_path__ (Required) save path for result. 
* __--config_path__ (Required) file of model hyper-parameters, can be stored in config file. 
* __--spm_model_path__ (Required) the path of model tokenizer. 
* __--batch_size__ (Optional) default by 1. suggestion: consistent with the input. 
* __--seq_length__ (Optional) default by 128. total length of generated content, equal to the length of input and generated sentence. 
* __--use_int8__ (Optional) default by False. whether use int8 to inference. 
* __--top_k__ (Optional) default by 40. 
* __--top_p__ (Optional) default by 0.95. 
* __--temperature__ (Optional) default by 0.8. 
* __--repetition_penalty_range__ (Optional) default by 1024. 
* __--repetition_penalty_slope__ (Optional) default by 0. 
* __--repetition_penalty__ (Optional) default by 1.15. 

<br> 

### Quick Start 
#### FP16/Int8 Inference 
fp16 inference： 
```commandline
python llama_infer.py --test_path ./prompts.txt --prediction_path ./result.txt  \
                      --load_model_path xxx.bin \
                      --config_path ./config/llama_7b_config.json \
                      --spm_model_path ./tokenizer.model
``` 


int8 inference: 
```commandline
python llama_infer.py --test_path ./prompts.txt --prediction_path ./result.txt  \
                      --load_model_path xxx.bin --use_int8 \
                      --config_path ./config/llama_7b_config.json \
                      --spm_model_path ./tokenizer.model
``` 

#### Microservices deployment 
need to install flask
```commandline
pip install flask 
python llama_server.py --load_model_path xxxx.bin \
                       --config_path config.json \
                       --spm_model_path tokenizer.model
```


<br>

#### Multi-GPU Inference 
need to install tensor_parallel
world_size = the number of gpu（gpu id start from 0.）
```commandline
pip install tensor_parallel
python llama_infer_tp.py --test_path ./prompts.txt --prediction_path ./result.txt \
                         --load_model_path xxxx.bin \
                         --config_path config.json \
                         --spm_model_path tokenizer.model \
                         --world_size 2
```
<br>