import argparse
from utils import load_hyperparam, convert_normal_parameter_to_int8, load_model
from model.tokenize import Tokenizer
from model.llama import *
from generate import LmGeneration


def multi_round_chat(args, lm_generation, keep_length_ratio=0.5):
    users = []
    answers = []
    while True:
        user_input = input("User: ")
        if user_input == 'clear':
            users = []
            answers = []
            print("开启新的一轮聊天/Start a new round of chat:")
            continue

        if user_input == 'exit':
            break

        input_str = ''
        for user, ans in zip(users, answers):
            input_str += 'User: ' + user + '\nBot: ' + ans + '\n'
        input_str += 'User: ' + user_input + '\nBot: '
        if len(input_str) >= int(keep_length_ratio * args.seq_length):
            input_str = input_str[len(input_str) - int(keep_length_ratio * args.seq_length):]
        answer = lm_generation.generate(args, [input_str], cut_off='User:', cut_off_times=1)[0]
        answer = answer[len(input_str):]
        print("ChatLLaMa: " + answer.replace('User:', ''))
        users.append(user_input.rstrip(' ').rstrip('\n'))
        answers.append(answer.replace('User:', '').rstrip(' ').rstrip('\n'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path of the input model.")
    parser.add_argument("--prediction_path", type=str, default=None,
                        help="Path of the prediction file.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path of the config file.")
    parser.add_argument("--seq_length", type=int, default=2048,
                        help="Sequence length.")
    parser.add_argument("--world_size", type=int, default=1,
                        help="the number of gpus.")
    parser.add_argument("--keep_length_ratio", type=float, default=0.5)
    parser.add_argument("--use_int8", action="store_true")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--repetition_penalty_range", type=int, default=1024)
    parser.add_argument("--repetition_penalty_slope", type=float, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)

    parser.add_argument("--spm_model_path", default=None, type=str,
                        help="Path of the sentence piece model.")

    args = parser.parse_args()

    args = load_hyperparam(args)
    args.batch_size = 1

    args.tokenizer = Tokenizer(model_path=args.spm_model_path)
    args.vocab_size = args.tokenizer.sp_model.vocab_size()

    torch.set_default_tensor_type(torch.HalfTensor)
    model = LLaMa(args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model = load_model(model, args.load_model_path)

    model.eval()
    # use multi-gpu tensor parallel
    if args.world_size > 1:
        import tensor_parallel as tp
        gpus = ["cuda:" + str(i) for i in range(args.world_size)]
        if args.use_int8:
            model = tp.tensor_parallel(model, gpus, delay_init=True)
            model = convert_normal_parameter_to_int8(model)
        else:
            model = tp.tensor_parallel(model, gpus)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    lm_generation = LmGeneration(model, args.tokenizer)
    multi_round_chat(args, lm_generation, args.keep_length_ratio)