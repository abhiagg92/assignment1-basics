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
        )
        self._model.to(self._device)
        self._model.eval()

        self._tokenizer = BPETokenizer.from_files(
            config.vocab_filepath,
            config.merges_filepath,
            config.special_tokens
        )

    @torch.no_grad()
    def decode(self, prompt: str, max_tokens: int | None=None):
        decoded_prompt = self._tokenizer.encode(prompt)
        decoded_prompt = torch.Tensor(decoded_prompt, device=self._device)
        generated_tokens = 0
        while True:
            if max_tokens and generated_tokens == max_tokens:
                break
            output = self._model(decoded_prompt.unsqueeze(0))
            last_pred_dist = softmax(output[0, -1]/self._config.temp, dim=-1)
            
            sorted_pred, indices = last_pred_dist.sort(descending=True)
            idx = torch.searchsorted(sorted_pred, self._config.top_p, right=False)
            topp_logits = last_pred_dist[indices[:idx]]
            
            dist = torch.distributions.Categorical(logits=topp_logits)
            sampled_idx = dist.sample()
            sampled_token = indices[sampled_idx]
            
            if sampled_token.item() == 0:
                break
            decoded_prompt = torch.cat((decoded_prompt, sampled_token))
            generated_tokens += 1