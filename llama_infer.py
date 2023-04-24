import argparse
from utils import load_hyperparam, convert_normal_parameter_to_int8
from model.tokenize import Tokenizer
from model.llama import *
from generate import LmGeneration


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
    parser.add_argument("--world_size", type=int, default=1,
                        help="the number of gpus.")
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

    args = parser.parse_args()

    args = load_hyperparam(args)

    args.tokenizer = Tokenizer(model_path=args.spm_model_path)
    args.vocab_size = args.tokenizer.sp_model.vocab_size()

    torch.set_default_tensor_type(torch.HalfTensor)
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

    model.eval()
    # use multi-gpu tensor parallel
    if args.world_size > 1:
        import tensor_parallel as tp
        gpus = ["cuda:" + str(i) for i in range(args.world_size)]
        if args.use_int8:
            import bitsandbytes as bnb
            model = tp.tensor_parallel(model, gpus)
            for name, parameter in model.named_parameters():
                print(name)
                print(parameter)
            exit()
            # model = convert_normal_parameter_to_int8(model)
            # print(model)
            # for key, value in model.named_parameters():
            #     print(key)
            #     print(value)
            # exit()

        else:
            model = tp.tensor_parallel(model, gpus)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    lm_generation = LmGeneration(model, args.tokenizer)
    prompts = []
    with open(args.test_path, 'r', encoding='utf-8') as f:
        for line in f:
            prompts.append(line)
    with torch.no_grad():
        result = lm_generation.generate(args, prompts)

    with open(args.prediction_path, 'w', encoding='utf-8') as f:
        for res in result:
            f.write(res + '\n')
            f.write('\n')