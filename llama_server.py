import argparse
import torch
from utils import load_hyperparam
from model.tokenize import Tokenizer
from model.llama import *
from generate import LmGeneration
from flask import Flask, request
import json

app = Flask(__name__)
args = None
lm_generation = None
if torch.cuda.is_available():
    torch.cuda.set_device(0)


def init_model():
    global args
    global lm_generation
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
        model = tp.tensor_parallel(model, gpus)
    else:
        # torch.cuda.set_device(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    lm_generation = LmGeneration(model, args.tokenizer)


@app.route("/chat", methods=['POST'])
def chat():
    question = request.json.get("question")
    if isinstance(question, str):
        question = [question, ]
    try:
        with torch.no_grad():
            answer = lm_generation.generate(args, question)
        status = 'success'
    except Exception:
        answer = ''
        status = 'error'
    return json.dumps({'answer': answer, 'status': status}, ensure_ascii=False)


if __name__ == '__main__':
    init_model()
    app.run(host='127.0.0.1', port=8888, debug=False)
