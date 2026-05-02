import torch

from cs336_basics.models import TransformerLM
from cs336_basics.config import DecoderConfig
from cs336_basics.tokenizers import BPETokenizer
from cs336_basics.models import softmax


class Decoder:
    def __init__(self, config: DecoderConfig):
        self._config = config
        self._device = torch.device("cuda")
        
        self._model = TransformerLM(
            vocab_size=config.vocab_size,
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            theta=config.rope_theta,
            context_length=config.context_length,
            device=self._device
        )
        self._model.to(self._device)
        self._model.eval()

        state = torch.load(config.ckpt_path)
        self._model.load_state_dict(state["model"])

        self._tokenizer = BPETokenizer.from_files(
            config.vocab_filepath,
            config.merges_filepath,
            config.special_tokens
        )

    @torch.no_grad()
    def decode(self, prompt: str, max_tokens: int | None=None):
        decoded_prompt = self._tokenizer.encode(prompt)[-self._config.context_length:]
        decoded_prompt = torch.tensor(decoded_prompt, device=self._device)
        generated_tokens_count = 0
        generated_tokens = []
        while True:
            if max_tokens and generated_tokens_count == max_tokens:
                break
            output = self._model(decoded_prompt.unsqueeze(0))
            last_pred_dist = softmax(output[0, -1]/self._config.temp, dim=-1)
            
            sorted_pred, indices = last_pred_dist.sort(descending=True)
            cumsum = sorted_pred.cumsum(dim=0)
            idx = torch.searchsorted(cumsum, self._config.top_p, right=False)
            topp_logits = last_pred_dist[indices[:idx+1]]
            
            dist = torch.distributions.Categorical(logits=topp_logits)
            sampled_idx = dist.sample()
            sampled_token = indices[sampled_idx]
            
            if sampled_token.item() == 0:
                break
            
            if len(decoded_prompt) >= self._config.context_length:
                decoded_prompt = torch.cat((decoded_prompt[1:], sampled_token))
            else:
                decoded_prompt = torch.cat((decoded_prompt, sampled_token.unsqueeze(0)))
            generated_tokens_count += 1
            generated_tokens.append(sampled_token.item())
        
        return self._tokenizer.decode(generated_tokens)