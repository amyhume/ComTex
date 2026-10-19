import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.spatial.distance import cosine
import re 
import pylzma
import os
import string
from comtex.utils import *

# ---------------------------
# LM wrapper class
# ---------------------------

class LMHeadModel:
    def __init__(self, model_name, device, max_ctx = None):
        self.device = device
        self.model_name = model_name
        
        #1. Loading Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #2. Loading Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
        
        self.model.eval()
        
        #setting context lim
        self.max_ctx = self.model.config.max_position_embeddings if max_ctx is None else max_ctx
        
        #context state
        self.context_tids = []
        self.context_word_lengths = []
        self.context_words = []
        #model state 
        self.past_key_values = None
        self.last_logits = None
        
    #function rebuilding cache from context
    @torch.no_grad()
    def rebuild_cache_from_ctx(self):

        if len(self.context_tids) == 0:
            self.past_key_values = None
            self.last_logits = None
            return
        
        input_ids = torch.tensor([self.context_tids], device=self.device)
        out = self.model(
            input_ids = input_ids,
            use_cache=True,
            output_hidden_states=False
        )
        self.past_key_values = out.past_key_values
        self.last_logits = out.logits[:, -1, :]
        
    #function to compute token level surprisal 
    def compute_token_surprisal(self, tid):
        if self.last_logits is None:
            return np.nan, np.nan
        log_probs = torch.log_softmax(self.last_logits, dim=1)
        
        #surprisal
        log_p = log_probs[0, tid].item()
        token_surprisal = -log_p / np.log(2)
        #entropy
        probs = torch.exp(log_probs)
        token_entropy = -(probs * log_probs).sum().item() / np.log(2)
        
        return token_surprisal, token_entropy, probs
    
    #This function processes a token given the self context values etc., and advances model forward 1 token
    def step(self, tid):
        """Process one token and update the model's state."""
        input_ids = torch.tensor([[tid]], device=self.device)
    
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=self.past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
    
        # Update model state
        self.last_logits = outputs.logits[:, -1, :]
        self.past_key_values = outputs.past_key_values
    
        # Update context
        self.context_tids.append(tid)
    
        return outputs.hidden_states


#Additional functions
def find_device():
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Using device: {device}")
    return device

def llm_process_words(words, lm):
    #WILL RETURN all of the following which will have lengths equivalent to initial word list
    
    #initializing processing specific outputs
    all_hidden_states = []
    all_sum_surprisal = []
    all_mean_surprisal = []
    all_entropy = []
    all_tokens = []

    with torch.no_grad():
        for idx, word in enumerate(words):  
            word = str(word)
            word_for_model = word if idx == 0 else " " + word
            token_ids = lm.tokenizer(word_for_model, add_special_tokens=False)['input_ids']
            tokens = lm.tokenizer.convert_ids_to_tokens(token_ids)
            all_tokens.append(tokens)
            
            #make room in context for incoming tokens
            incoming_word_length = len(token_ids)
            
            if len(lm.context_tids) + incoming_word_length > lm.max_ctx:
                #while the values left in context word lengths + incoming word length exceed context, keep removing words 
                n_remove = 0
                while sum(lm.context_word_lengths[n_remove:]) + incoming_word_length > lm.max_ctx:
                    n_remove += 1
    
                #how many total tokens to remove 
                n_tokens_remove = sum(lm.context_word_lengths[:n_remove])
                lm.context_words = lm.context_words[n_remove:]
                lm.context_word_lengths = lm.context_word_lengths[n_remove:]
                lm.context_tids = lm.context_tids[n_tokens_remove:]
                #rebuild cache if context changed 
                if n_remove > 0:
                    lm.rebuild_cache_from_ctx()
            
            #initialise word surprisal as 0, or np.nan for first token 
            word_sum_surprisal = np.nan if lm.last_logits is None else 0.0
            word_mean_surprisal = np.nan if lm.last_logits is None else 0.0
            word_entropy = np.nan
            
            
            for i, tid in enumerate(token_ids):
                
                #calculate surprisal before processing the incoming token
                if lm.last_logits is not None:
                    token_surprisal, token_entropy, probs = lm.compute_token_surprisal(tid)
                    word_sum_surprisal += token_surprisal 
                    if i == 0:
                        word_entropy = token_entropy
    
                #have the model process that token and advance forward 
                hidden_states = lm.step(tid)
            
            word_mean_surprisal = word_sum_surprisal / incoming_word_length
            lm.context_word_lengths.append(len(token_ids))
            lm.context_words.append(word)
            
            #hidden state for last token in the word 
            #defaults to last token in word because hidden_states will = those of the last tid in token_ids
            word_hidden_states = [
                h[0, -1, :].detach().cpu().numpy()
                for h in hidden_states
            ]
            
            all_hidden_states.append(word_hidden_states)
            all_sum_surprisal.append(word_sum_surprisal)
            all_mean_surprisal.append(word_mean_surprisal)
            all_entropy.append(word_entropy)
        
            if idx % 20 == 0:
                print(f"Processed {idx+1}/{len(words)}")
                
        all_hidden_states = np.array(all_hidden_states, dtype=np.float32)
        return all_hidden_states, all_sum_surprisal, all_mean_surprisal, all_entropy, all_tokens
    
def emb_cosine(embeddings):
    from scipy.spatial.distance import cosine
    cos_values = [np.nan]

    for i in range(1, len(embeddings)):
        v_prev = embeddings[i-1]
        v_curr = embeddings[i]

        cos = cosine(v_prev, v_curr)
        cos_values.append(cos)

    return cos_values

def normalize_word(w):
    import re
    import string
    
    # skip NaN / non-string
    if not isinstance(w, str):
        return ""

    w = w.lower()
    # remove punctuation
    w = w.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    w = re.sub(r"\s+", " ", w)
    return w.strip()

def com(s):
    import pylzma
    compressed = pylzma.compress(s, eos=0)
    return len(compressed)

def Kolmo(s):
    import pylzma
    # Estimate complexity
    reduce = com("a" * len(s))
    compressed = pylzma.compress(s, eos=0)
    return len(compressed) - reduce

def compute_cum_kolmo(words):
    ctx_words = []
    kolmo_complexity = []
    
    for idx, word in enumerate(words):
        norm_word = normalize_word(word)
        ctx_words.append(norm_word)
        kolmo_complexity.append(Kolmo(" ".join(ctx_words)))
        
    return kolmo_complexity


#FUNCTION FOR PROCESSING TRANSCRIPT
def process_transcript(input_csv, output_folder, model_name, device, max_dur = None):
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    
    print("================================")
    print("1. Reading csv")
    
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"SKIPPED {os.path.basename(input_csv)} - empty transcription csv")
        return
    
    n_rows = len(df)
    dur = max(df['offset'])
    mean_confidence = df['confidence'].mean()
    
    #skipping based on confidence threshold / max duration
    if mean_confidence < 0.3:
        print(f"SKIPPED {os.path.basename(input_csv)} - mean confidence: {mean_confidence}")
        return
    if max_dur is not None:
        if dur > max_dur:
            print(f"SKIPPED {os.path.basename(input_csv)} - file too long: {dur:.3f}s")
            return
    
    #INITIALISING MODEL
    print("================================")
    print("2. Initialising LLM")
    lm = LMHeadModel(model_name, device)
    
    #model info
    hidden_size = lm.model.config.hidden_size
    n_layers = lm.model.config.num_hidden_layers
    
    words = df['text'].values
    
    #LLM PROCESSING
    print("================================")
    print("3. Computing hidden states, surprisals and entropies")
    hidden_states, sum_surprisals, mean_surprisals, entropies, tokens = llm_process_words(words, lm)
    
    #appending to dataset
    df['tokens'] = tokens
    df['sum_surprisal'] = sum_surprisals
    df['mean_surprisal'] = mean_surprisals
    df['entropy'] = entropies
    
    print("================================")
    print("4. Computing kolmogorov complexity")
    kolmo_complexity = compute_cum_kolmo(words)
    df['complexity'] = kolmo_complexity
    df['inforate'] = df['complexity'].diff()
        
    #cosine from embeddings
    #hidden_states is in shape (word_n, layer_n, dimensions)
    print("================================")
    print("5. Computing cosine distance of layer embeddings")
    layer_dict = {f"layer_{li}": hidden_states[:, li, :] for li in range(hidden_states.shape[1])}
    
    for li, layer in enumerate(layer_dict):
        col_name = f"cosine_l{str(li)}"
        cos_dist = emb_cosine(layer_dict[layer])
        df[col_name] = cos_dist
        
    #SAVING
    print("================================")
    print("6. Saving files")
    out_path = f"{output_folder}/{os.path.basename(input_csv)}"
    df.to_csv(out_path, index=False)
    print(f"saved {out_path}")
    np.savez_compressed(f"{output_folder}/{base_name}.npz", **layer_dict)  #EDIT PATH
    print(f"saved {output_folder}/{base_name}.npz")
    
    return df

def run_transcript_folder(folder, model_name, device, max_dur=None,out_dir=None):
    safe_model = model_name.replace('/', "_")
    
    if out_dir is None:
        out_dir = os.path.join(folder, "output", safe_model)
    os.makedirs(out_dir, exist_ok=True)
    
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    
    for fname in files:
        fid = os.path.splitext(fname)[0]
        in_path = os.path.join(folder, fname)
        
        if os.path.exists(f"{out_dir}/{fname}"):
            print(f"✓ Skipping {fname} (already processed)")
            continue
        
        print(f"\n=== Processing {fname} ===")
        
        process_transcript(in_path, out_dir, model_name, device, max_dur)
        
        
#FUNCTIONS FOR PROCESSING FRAME CAPTIONS
def llm_process_captions(captions, lm):
    """ 
    Takes a list of frame captions, then for each one it goes word by word to calculate word level surprisal
    at frame level, returns mean surprisal and entropy
    also returns the hidden states for the last word of each frame 
    
    """
    mean_caption_sum_surprisals = []
    mean_caption_mean_surprisals = []
    mean_caption_entropies = []
    caption_hidden_states = []
    
    for idx, caption in enumerate(captions):
        print(f"=== Caption {idx}/{len(captions)}")
        words = caption.split(" ")
        hidden_states, sum_surprisals, mean_surprisals, entropies, tokens = llm_process_words(words, lm)
        
        mean_caption_sum_surprisals.append(np.nanmean(sum_surprisals))
        mean_caption_mean_surprisals.append(np.nanmean(mean_surprisals))
        mean_caption_entropies.append(np.nanmean(entropies))
        caption_hidden_states.append(hidden_states[-1])
    
    caption_hidden_states = np.array(caption_hidden_states, dtype=np.float32)
    
    return caption_hidden_states, mean_caption_sum_surprisals, mean_caption_mean_surprisals, mean_caption_entropies

def process_frame_captions(input_csv, output_folder, model_name, device, max_dur=None):
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    
    print("================================")
    print("1. Reading csv")
    
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"SKIPPED {os.path.basename(input_csv)} - empty frame caption csv")
        return
    
    n_rows = len(df)
    dur = n_rows
    
    if max_dur is not None:
        if dur > max_dur:
            print(f"SKIPPED {os.path.basename(input_csv)} - file too long: {dur:.3f}s")
            return
        
    #INITIALISING MODEL
    print("================================")
    print("2. Initialising LLM")
    lm = LMHeadModel(model_name, device)
    
    #model info
    hidden_size = lm.model.config.hidden_size
    n_layers = lm.model.config.num_hidden_layers
    
    
    captions = df['description'].values
    #LLM PROCESSING
    print("================================")
    print("3. Computing hidden states, surprisals and entropies")
    hidden_states, sum_surprisals, mean_surprisals, entropies = llm_process_captions(captions, lm)
    
    #appending to dataset
    df['mean_sum_surprisal'] = sum_surprisals
    df['mean_mean_surprisal'] = mean_surprisals
    df['mean_entropy'] = entropies
    
    print("================================")
    print("4. Computing kolmogorov complexity")
    kolmo_complexity = compute_cum_kolmo(captions)
    df['complexity'] = kolmo_complexity
    df['inforate'] = df['complexity'].diff()
    
    #cosine from embeddings
    #hidden_states is in shape (word_n, layer_n, dimensions)
    print("================================")
    print("5. Computing cosine distance of layer embeddings")
    layer_dict = {f"layer_{li}": hidden_states[:, li, :] for li in range(hidden_states.shape[1])}
    
    for li, layer in enumerate(layer_dict):
        col_name = f"cosine_l{str(li)}"
        cos_dist = emb_cosine(layer_dict[layer])
        df[col_name] = cos_dist
    
    #SAVING
    print("================================")
    print("6. Saving files")
    out_path = f"{output_folder}/{os.path.basename(input_csv)}"
    df.to_csv(out_path, index=False)
    print(f"saved {out_path}")
    np.savez_compressed(f"{output_folder}/{base_name}.npz", **layer_dict)  #EDIT PATH
    print(f"saved {output_folder}/{base_name}.npz")
        
    
def run_caption_folder(folder, model_name, device, max_dur=None,out_dir=None):
    safe_model = model_name.replace('/', "_")
    
    if out_dir is None:
        out_dir = os.path.join(folder, "output", safe_model)
    os.makedirs(out_dir, exist_ok=True)
    
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    
    for fname in files:
        fid = os.path.splitext(fname)[0]
        in_path = os.path.join(folder, fname)
        
        if os.path.exists(f"{out_dir}/{fname}"):
            print(f"✓ Skipping {fname} (already processed)")
            continue
        
        print(f"\n=== Processing {fname} ===")
        
        process_frame_captions(in_path, out_dir, model_name, device, max_dur)