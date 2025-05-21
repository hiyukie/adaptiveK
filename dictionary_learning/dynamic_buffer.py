import torch as t
from nnsight import LanguageModel
import gc
from tqdm import tqdm

from .config import DEBUG

if DEBUG:
    tracer_kwargs = {'scan': True, 'validate': True}
else:
    tracer_kwargs = {'scan': False, 'validate': False}

class DynamicTopKBuffer:
    
    def __init__(
        self,
        data,  # text generator
        model: LanguageModel,
        submodule,
        n_ctxs=3e4,
        ctx_len=1024,
        refresh_batch_size=512,  # refresh batch size
        out_batch_size=8192,  # output batch size
        io='out',
        d_submodule=None,
        device='cpu',
        remove_bos: bool = False,
        data_factory=None
    ):
        self.data_factory = data_factory
        self.data_cache = None
        
        if io not in ['in', 'out']:
            raise ValueError("io must be either 'in' or 'out'")

        if d_submodule is None:
            try:
                if io == 'in':
                    d_submodule = submodule.in_features
                else:
                    d_submodule = submodule.out_features
            except:
                raise ValueError("d_submodule cannot be inferred and must be specified directly")
        
        self.data = data
        self.model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.remove_bos = remove_bos
        
        self.activation_buffer_size = int(n_ctxs)
        
        # Create activation buffer
        self.activations = t.empty(0, d_submodule, device=device, dtype=model.dtype)
        self.read = t.zeros(0).bool()
        
        # Track current data position
        self.data_position = 0
        
        self._cache_data()
        
        # Cache completion flag
        self.cached_complete = False
    
    def _cache_data(self):
        if self.data_cache is None:
            try:
                if hasattr(self.data, '__iter__') and not hasattr(self.data, '__len__'):
                    self.data_cache = list(self.data)
                    print(f"[Dynamic TopK buffer] Cached {len(self.data_cache)} text samples")
                else:
                    self.data_cache = self.data
                    print(f"[Dynamic TopK buffer] Using existing data source, size: {len(self.data_cache) if hasattr(self.data_cache, '__len__') else 'unknown'}")
            except Exception as e:
                print(f"[Dynamic TopK buffer] Error caching data: {e}")
                import traceback
                print(traceback.format_exc())
    
    def text_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.refresh_batch_size
            
        try:
            if self.data_cache is None:
                self._cache_data()

            if self.data_cache is None:
                # Get directly from data source
                return [next(self.data) for _ in range(batch_size)]
            
            batch = []
            end_pos = min(self.data_position + batch_size, len(self.data_cache))
            
            if self.data_position >= len(self.data_cache):
                print(f"[Dynamic TopK buffer] Data exhausted (position {self.data_position}), resetting position")
                self.data_position = 0
                end_pos = min(batch_size, len(self.data_cache))
            
            batch = self.data_cache[self.data_position:end_pos]
            self.data_position = end_pos
            
            if len(batch) == 0:
                print(f"[Dynamic TopK buffer] Warning: Got empty batch")
                raise StopIteration("No data available")
            
            print(f"[Dynamic TopK buffer] Extracted {len(batch)} samples from position {self.data_position-len(batch)}")
            return batch
            
        except Exception as e:
            if self.data_factory is not None:
                self.data = self.data_factory()
                self.data_cache = None  # Clear cache for reload
                self._cache_data()  # Recache
                self.data_position = 0
                # Recursive call to retry
                return self.text_batch(batch_size)
            else:
                print(f"[Dynamic TopK buffer] Failed to get text batch: {e}")
                import traceback
                print(traceback.format_exc())
                raise
    
    def tokenized_batch(self, batch_size=None):
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True
        )
    
    def refresh(self):
        if hasattr(self, 'activations') and len(self.activations) == self.activation_buffer_size and hasattr(self, 'cached_complete') and self.cached_complete:
            self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)
            return

        gc.collect()
        t.cuda.empty_cache()
        
        read_count = self.read.sum().item() if hasattr(self, 'read') and len(self.read) > 0 else 0
        unread_count = (~self.read).sum().item() if hasattr(self, 'read') and len(self.read) > 0 else 0
        
        print(f"[Dynamic TopK buffer] Starting refresh - Read samples: {read_count}, Unread samples: {unread_count}, Total capacity: {self.activation_buffer_size}")
        
        if self.data_factory is not None and self.data_cache is None:
            self.data = self.data_factory()
            self._cache_data() 
            self.data_position = 0
        
        if hasattr(self, 'read') and len(self.read) > 0:
            self.activations = self.activations[~self.read]
        
        current_idx = len(self.activations)
        new_activations = t.empty(self.activation_buffer_size, self.d_submodule, device=self.device, dtype=self.model.dtype)

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        batch_count = 0
        max_attempts = 10000000
        
        expected_batches = self.activation_buffer_size // 64 + (1 if self.activation_buffer_size % 64 > 0 else 0)
        print(f"[Dynamic TopK buffer] Expecting approximately {expected_batches} batches to process {self.activation_buffer_size} samples")
        
        pbar = tqdm(total=self.activation_buffer_size, desc="Collecting context representation vectors")
        pbar.update(current_idx)
        
        while current_idx < self.activation_buffer_size and batch_count < max_attempts:
            batch_count += 1
            print(f"[Dynamic TopK buffer] Processing batch {batch_count}, currently collected {current_idx}/{self.activation_buffer_size} context representations")
            
            try:
                with t.no_grad():
                    text_batch = self.text_batch()
                    batch_size = len(text_batch)
                    print(f"[Dynamic TopK buffer] Text batch size: {batch_size}")
                    
                    with self.model.trace(
                        text_batch,
                        **tracer_kwargs,
                        invoker_args={"truncation": True, "max_length": self.ctx_len},
                    ):
                        if self.io == "in":
                            hidden_states = self.submodule.inputs[0].save()
                        else:
                            hidden_states = self.submodule.output.save()
                        input = self.model.inputs.save()
                        self.submodule.output.stop()
                        
                attn_mask = input.value[1]["attention_mask"]
                hidden_states = hidden_states.value
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                    
                if self.remove_bos:
                    hidden_states = hidden_states[:, 1:, :]
                    attn_mask = attn_mask[:, 1:]
                
                context_vectors = []
                for i in range(hidden_states.shape[0]):
                    seq_len = attn_mask[i].sum().item()
                    if seq_len > 0:
                        last_token_idx = seq_len - 1
                        context_vectors.append(hidden_states[i, last_token_idx])
                
                if context_vectors:
                    context_vectors = t.stack(context_vectors)
                    
                    # Check for zero vectors
                    zero_vectors = (t.norm(context_vectors, dim=1) == 0).sum().item()
                    if zero_vectors > 0:
                        print(f"[Dynamic TopK buffer warning] Found {zero_vectors} zero vectors!")
                    
                    non_zero_mask = t.norm(context_vectors, dim=1) > 0
                    if non_zero_mask.sum().item() < len(context_vectors):
                        print(f"[Dynamic TopK buffer warning] Filtered out {len(context_vectors) - non_zero_mask.sum().item()} zero vectors")
                        context_vectors = context_vectors[non_zero_mask]
                    
                    print(f"[Dynamic TopK buffer] Extracted {len(context_vectors)} context representation vectors in this batch")
                    if len(context_vectors) > 0:
                        print(f"[Dynamic TopK buffer] Vector stats: min={context_vectors.min().item():.4f}, max={context_vectors.max().item():.4f}")
                        print(f"[Dynamic TopK buffer] Vector L2 norm mean: {t.norm(context_vectors, dim=1).mean().item():.4f}")
                    
                    remaining_space = self.activation_buffer_size - current_idx
                    available_count = min(len(context_vectors), remaining_space)
                    context_vectors = context_vectors[:available_count]
                    
                    self.activations[current_idx:current_idx+available_count] = context_vectors.to(self.device)
                    current_idx += available_count
                    pbar.update(available_count)
                    
                    print(f"[Dynamic TopK buffer] Processed batch {batch_count}, currently collected {current_idx}/{self.activation_buffer_size} context representations")
            
            except Exception as e:
                print(f"[Dynamic TopK buffer] Error processing batch: {e}")
                import traceback
                print(traceback.format_exc())
                continue

        pbar.close()
        
        # Check if enough samples were collected
        if current_idx < self.activation_buffer_size:
            print(f"[Dynamic TopK buffer warning] Could not collect enough samples! Current: {current_idx}/{self.activation_buffer_size}")
            print(f"[Dynamic TopK buffer] Attempted batches: {batch_count}/{max_attempts}")
            self.activations = self.activations[:current_idx]
            print(f"[Dynamic TopK buffer] Adjusted activation buffer size to actual collected samples: {current_idx}")

        print(f"[Dynamic TopK buffer] Refresh complete, collected a total of {current_idx} context representation vectors, processed {batch_count} text batches")
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

        self.cached_complete = True
    
    def __iter__(self):
        if hasattr(self, 'read') and len(self.read) > 0:
            self.read = t.zeros_like(self.read)
        return self
        
    def __next__(self):
        if not hasattr(self, 'read') or len(self.read) == 0 or self.read.all():
            self.refresh()
            
        batch_indices = (~self.read).nonzero().view(-1)
        if len(batch_indices) == 0:
            raise StopIteration
            
        batch_size = min(self.out_batch_size, len(batch_indices))
        indices = batch_indices[:batch_size]
        self.read[indices] = True
        
        return self.activations[indices]
        
    @property
    def config(self):
        return {
            'd_submodule': self.d_submodule,
            'io': self.io,
            'n_ctxs': self.n_ctxs,
            'ctx_len': self.ctx_len,
            'refresh_batch_size': self.refresh_batch_size,
            'out_batch_size': self.out_batch_size,
            'device': self.device
        }