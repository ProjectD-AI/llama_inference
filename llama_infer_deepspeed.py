import argparse
from utils import load_hyperparam, _load_state_dict_into_model
from model.tokenize import Tokenizer
from model.llama import *
from generate import LmGeneration
import deepspeed
import torch.distributed as dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path of the input model.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--prediction_path", type=str, required=True,
                        help="Path of the prediction file.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path of the config file.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--use_int8", action="store_true")
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty_range", type=int, default=1024)
    parser.add_argument("--repetition_penalty_slope", type=float, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)

    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")

    # deepspeed 配置
    parser.add_argument("--deepspeed", action="store_true",
                        help=".")
    parser.add_argument("--enable_zero3", action="store_true",
                        help=".")
    parser.add_argument("--deepspeed_config", default="config/deepspeed_config.json", type=str,
                        help=".")
    parser.add_argument("--deepspeed_checkpoint_activations", action='store_true',
                        help="Checkpoint activation to allow for training with larger models, sequences, and batch sizes.")
    parser.add_argument("--deepspeed_checkpoint_layers_num", type=int, default=1,
                        help="chunk size (number of layers) for checkpointing.")
    parser.add_argument("--local_rank", type=int, required=False)

    args = parser.parse_args()

    args = load_hyperparam(args)

    args.tokenizer = Tokenizer(model_path=args.spm_model_path)
    args.vocab_size = args.tokenizer.sp_model.vocab_size()

    torch.set_default_tensor_type(torch.HalfTensor)

    if args.enable_zero3:
        with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
            model = LLaMa(args)
            model = _load_state_dict_into_model(model, args.load_model_path)
    else:
        model = LLaMa(args)
        torch.set_default_tensor_type(torch.FloatTensor)
        checkpoint = torch.load(args.load_model_path, map_location='cpu')
        for parameter_name, parameter in model.named_parameters():
            if 'target' in parameter_name:
                parameter.data = checkpoint['target.lm.output_layer.weight']
            elif 'embedding' in parameter_name:
                parameter.data = checkpoint['embedding.word.embedding.weight']
            else:
                parameter.data = checkpoint[parameter_name]
            parameter.requires_grad = False
        del checkpoint

    deepspeed.init_distributed()
    model = deepspeed.initialize(model=model, config_params=args.deepspeed_config)[0]
    rank = dist.get_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    lm_generation = LmGeneration(model, args.tokenizer)
    prompts = []
    with open(args.test_path, 'r', encoding='utf-8') as f:
        for line in f:
            prompts.append(line)
    with torch.no_grad():
        result = lm_generation.generate(args, prompts)

    if rank == 0:
        with open(args.prediction_path, 'w', encoding='utf-8') as f:
            for res in result:
                f.write(res + '\n')
                f.write('\n')