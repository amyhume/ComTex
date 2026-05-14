import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import cosine
import re 
import pylzma
import os
import string

# ---------------------------
# LM wrapper class
# ---------------------------

class LMHeadModel:
    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name

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

def log_step(step, msg):
    print(f"\n[{step}] {msg}")
    print("=" * 50)

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

#takes embeddings and computes cosine distance time series
def emb_cosine(embeddings):
    cos_values = [np.nan]

    for i in range(1, len(embeddings)):
        v_prev = embeddings[i-1]
        v_curr = embeddings[i]

        cos = cosine(v_prev, v_curr)
        cos_values.append(cos)

    return cos_values

def normalize_word(w):
    # skip NaN / non-string
    if not isinstance(w, str):
        return ""

    w = w.lower()
    # remove punctuation
    w = w.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    w = re.sub(r"\s+", " ", w)
    return w.strip()

### KOLMO PART ###
def com(s):
    compressed = pylzma.compress(s, eos=0)
    return len(compressed)

def Kolmo(s):
    # Estimate complexity
    reduce = com("a" * len(s))
    compressed = pylzma.compress(s, eos=0)
    return len(compressed) - reduce

#this function takes the csv, runs the model, saves the embeddings and returns the csv with complexity values appended 
def compute_surprisal_for_csv(input_csv_path, output_folder, model=None, confidence_threshold=0.3, duration_threshold=10800):
    log_step(1, f"Processing file: {input_csv_path}")

    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        log_step('ERROR', f'Skipped {os.path.basename(input_csv_path)}, empty transcription csv')
        return
    
    base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    n_rows = len(df)

    dur = max(df['offset'])
    mean_confidence = df['confidence'].mean()

    log_step('', f"Successfully loaded {base_name}: mean_confidence={mean_confidence:.3f}")

    if mean_confidence < confidence_threshold:
        log_step('WARNING', f'Skipping {base_name}: mean_confidence={mean_confidence:.3f}')
        return 
    
    if dur > duration_threshold:
        log_step('WARNING', f'Skipping {base_name}, file too long: file_duration={dur:.3f}')
        return 

    #extracting model info 
    if model is None:
        print('LLM model is not defined or loaded. Check and try again')
        return
    
    model_name = model.model_name
    hidden_size = model.model.config.hidden_size
    n_layers = model.model.config.num_hidden_layers
    N = len(df)
    model.reset_cache()

    log_step(2, f"Starting llm processing of transcript")

    list_surprisal, list_entropy, list_complexity = [], [], []
    context_words = []

    layer_arrays = [np.zeros((N, hidden_size), dtype=np.float32) for _ in range(n_layers + 1)]

    for idx, word in enumerate(df["text"]):
        norm_word = normalize_word(word)
        context_words.append(norm_word)

        token_ids = model.tokenizer(
            norm_word,
            add_special_tokens=False
        ).input_ids

        # print(token_ids)
        #if word is not in models vocab, it will add an na value to csv instead and skip
        if len(token_ids) == 0:
            list_surprisal.append(np.nan)
            list_entropy.append(np.nan)
            list_complexity.append(np.nan)
            for li in range(n_layers + 1):
                layer_arrays[li][idx] = np.nan
            continue

        token_surprisal = 0.0
        last_hidden_states = None
        probs = None

        for tid in token_ids:
            if model.last_logits is not None:
                probs = torch.softmax(model.last_logits, dim=-1)
                p = probs[0, tid].item()
                token_surprisal += -np.log2(p)

            logits, hidden_states = model.step(tid)
            last_hidden_states = hidden_states
        
        #add representations to array
        for li in range(n_layers + 1):
            layer_arrays[li][idx] = last_hidden_states[li][0, -1].detach().cpu().numpy().astype(np.float32)

        # entropy from last token
        if probs is not None:
            probs_np = probs.detach().cpu().numpy().squeeze()
            entropy = -np.sum(probs_np * np.log2(probs_np + 1e-12))
        else:
            entropy = np.nan

        list_surprisal.append(token_surprisal)
        list_entropy.append(entropy)
        list_complexity.append(Kolmo(" ".join(context_words)))

        if idx % 20 == 0:
            print(f"Processed {idx+1}/{len(df)}")
            # print(context_words[-10:])

    #appending outputs to df
    df["Surprisal"] = list_surprisal
    df["Entropy"] = list_entropy
    df["Complexity"] = list_complexity
    df["InfoRate"] = df["Complexity"].diff()

    #doing cosine from embeddings
    layer_dict = {f"layer_{L}": layer_arrays[L] for L in range(n_layers + 1)}

    for li, layer in enumerate(layer_dict):
        col_name = f"cosine_l{str(li)}"
        cos_dist = emb_cosine(layer_dict[layer])
        df[col_name] = cos_dist

    df['transformer_model'] = model_name

    log_step(3, "Saving output...")
    #output final df
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, os.path.basename(input_csv_path))
    df.to_csv(out_path, index=False)
    np.savez_compressed(f"{output_folder}/{base_name}_embeddings.npz", **layer_dict)  #EDIT PATH
    print(f"Saved → {out_path}")

    return df

def run_folder_llm(folder, out_dir=None, model=None):
    if model is None:
        print('LLM model is not defined or loaded. Check and try again')
        return
    
    model_name = model.model_name
    hidden_size = model.model.config.hidden_size
    n_layers = model.model.config.num_hidden_layers

    if out_dir is None:
        out_dir = os.path.join(folder, 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    files = [f for f in sorted(os.listdir(folder)) if f.endswith('.csv')]
    log_step('', f'Processing folder: {folder} - {len(files)} files')
    print(f"Using model: {model_name}\n",
          f"Hidden size: {hidden_size}\n",
          f"Number of layers: {n_layers}")
    
    for fname in files:
        in_path = os.path.join(folder, fname)

        if os.path.exists(os.path.join(out_dir, fname)):
            print(f"✓ Skipping {fname} (already processed)")
            continue

        compute_surprisal_for_csv(in_path, out_dir, model=model)