import gradio as gr
import argparse
from utils import load_hyperparam, load_model, convert_normal_parameter_to_int8
from model.tokenize import Tokenizer
from model.llama import *
from generate import LmGeneration


args = None
lm_generation = None


def init_args():
    global args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path of the input model.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path of the config file.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--world_size", type=int, default=1,
                        help="the number of gpus.")
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

    args.tokenizer = Tokenizer(model_path=args.spm_model_path)
    args.vocab_size = args.tokenizer.sp_model.vocab_size()


def init_model():
    global lm_generation
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


def chat(prompt, top_k, temperature):
    args.top_k = int(top_k)
    args.temperature = temperature
    response = lm_generation.generate(args, [prompt])
    return response[0]


if __name__ == '__main__':
    init_args()
    init_model()
    demo = gr.Interface(
        fn=chat,
        inputs=["text", gr.Slider(1, 60, value=40, step=1), gr.Slider(0.1, 2.0, value=1.2, step=0.1)],
        outputs="text",
    )
    demo.launch()

