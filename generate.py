import torch
import torch.nn.functional as F


def apply_temperature(scores, tempt):
    scores = scores / tempt
    return scores


def apply_top_p(scores, top_p, filter_value=-float("Inf"), min_tokens_to_keep=1):
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def top_k_top_p_filtering(logits, top_k, top_p):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")
    return logits


def apply_advanced_repetition_penalty(
    input_ids, scores, penalty_range, penalty_slope, penalty
):
    penalty_range = int(penalty_range)
    clipped_penalty_range = min(input_ids.shape[-1], penalty_range)

    if penalty != 1.0:
        if penalty_range > 0:
            if clipped_penalty_range < input_ids.shape[1]:
                input_ids = input_ids[..., -clipped_penalty_range:]

            if penalty_slope != 0:
                _penalty = (
                    torch.arange(
                        penalty_range, dtype=scores.dtype, device=scores.device
                    )
                    / (penalty_range - 1)
                ) * 2.0 - 1
                _penalty = (penalty_slope * _penalty) / (
                    1 + torch.abs(_penalty) * (penalty_slope - 1)
                )
                _penalty = 1 + ((_penalty + 1) / 2).unsqueeze(0) * (penalty - 1)
                penalty = _penalty[..., -clipped_penalty_range:]

        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score <= 0, score * penalty, score / penalty)
        scores.scatter_(1, input_ids, score)

    return scores


class LmGeneration:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, args, prompts):
        batch = len(prompts)
        assert batch <= args.batch_size

        prompt_tokens = [args.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_len = min([len(x) for x in prompt_tokens])
        # max_prompt_len = max([len(x) for x in prompt_tokens])

        total_len = args.seq_length

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokens = torch.full((batch, total_len), self.tokenizer.pad_id).to(device).long()
        for idx, t in enumerate(prompt_tokens):
            tokens[idx, : len(t)] = torch.tensor(t).long()
        mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_len
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if args.temperature > 0:
                logits = top_k_top_p_filtering(logits, top_k=args.top_k, top_p=args.top_p)
                logits = apply_temperature(logits, args.temperature)
                logits = apply_advanced_repetition_penalty(
                    tokens[:, :cur_pos],
                    logits,
                    args.repetition_penalty_range,
                    args.repetition_penalty_slope,
                    args.repetition_penalty
                )
                scores = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(scores, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(
                mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
        print(tokens)

        decoder = []
        for i, t in enumerate(tokens.to_list()):
            t = t[: args.seq_length]
            try:
                t = t[:, t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoder.append(self.tokenizer.decode(t))
        print(decoder)

