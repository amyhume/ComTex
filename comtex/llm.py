import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------
# LM wrapper class
# ---------------------------

class LMHeadModel:
    def __init__(self, model_name, device):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        self.model.eval()

        self.max_ctx = int(getattr(self.model.config, "max_position_embeddings", 2048))
        self.ctx_token_ids = []
        # incremental state
        self.past_key_values = None
        self.last_hidden_states = None
        self.last_logits = None

    def reset_cache(self):
        self.past_key_values = None
        self.last_logits = None
        self.ctx_token_ids = []
    
    @torch.no_grad()
    def _rebuild_cache_from_ctx(self):
        """Recompute past_key_values / last_logits from ctx_token_ids (<= max_ctx-1)."""
        self.past_key_values = None
        self.last_logits = None

        if len(self.ctx_token_ids) == 0:
            return
        
        input_ids = torch.tensor([self.ctx_token_ids], device=self.device)
        out = self.model(
            input_ids = input_ids,
            use_cache=True,
            output_hidden_states=False
        )
        self.past_key_values = out.past_key_values
        self.last_logits = out.logits[:, -1, :] # distribution after last ctx token
 
    @torch.no_grad()
    def step(self, token_id: int):
        """
        Advance the model by ONE token using cached past_key_values.
        Returns:
          logits for next token
          hidden_states for this token
        """
        # NEW: if adding this token would exceed context, slide + rebuild
        if len(self.ctx_token_ids) >= (self.max_ctx - 1):
            self.ctx_token_ids = self.ctx_token_ids[-(self.max_ctx - 1):]
            self._rebuild_cache_from_ctx()

        input_ids = torch.tensor([[token_id]], device=self.device)

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=self.past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )

        # update cache
        self.past_key_values = outputs.past_key_values

        # logits for NEXT token
        logits = outputs.logits[:, -1, :]  # [1, vocab]
        self.last_logits = logits
        #new: append token to rolling context
        self.ctx_token_ids.append(token_id)
        # hidden states for CURRENT token
        hidden_states = outputs.hidden_states  # tuple(layer)[1, 1, dim]

        return logits, hidden_states


    @torch.no_grad()
    def step_tokens(self, input_ids):
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=self.past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        self.past_key_values = outputs.past_key_values
        self.last_logits = outputs.logits[:, -1]
        return outputs.hidden_states


    def surprisal_entropy_from_last_logits(self, token_id):
        probs = torch.softmax(self.last_logits, dim=-1)
        p = probs[0, token_id].item()
        surprisal = -np.log2(p) if p > 0 else np.nan
        entropy = -torch.sum(probs * torch.log2(probs)).item()
        return surprisal, entropy

    @torch.no_grad()
    def get_surprisal_entropy(self, word):
        """
        Compute surprisal from last_logits WITHOUT advancing cache
        """
        token_ids = self.tokenizer(
            word,
            add_special_tokens=False
        ).input_ids

        if self.last_logits is None:
            return np.nan, np.nan

        log_probs = torch.log_softmax(self.last_logits, dim=-1)

        total_logprob = 0.0
        for tid in token_ids:
            total_logprob += log_probs[0, tid].item()

        surprisal = -total_logprob / np.log(2)

        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum().item() / np.log(2)

        return surprisal, entropy

def load_llm(model_name='EleutherAI/pythia-1b', device=None):

    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    print(f"Using device: {device}")
    lm = LMHeadModel(model_name, device)
    return lm 