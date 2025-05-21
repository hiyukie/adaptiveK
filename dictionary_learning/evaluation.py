import torch as t
from collections import defaultdict

from .buffer import ActivationBuffer, NNsightActivationBuffer
from .dynamic_buffer import DynamicTopKBuffer
from nnsight import LanguageModel
from .config import DEBUG

def loss_recovered(
    text,  # a batch of text
    model: LanguageModel,  # an nnsight LanguageModel
    submodule,  # submodules of model
    dictionary,  # dictionaries for submodules
    max_len=None,  # max context length for loss recovered
    normalize_batch=False,  # normalize batch before passing through dictionary
    io="out",  # can be 'in', 'out', or 'in_and_out'
    tracer_args = {'use_cache': False, 'output_attentions': False}, # minimize cache during model trace.
):
    
    if max_len is None:
        invoker_args = {}
    else:
        invoker_args = {"truncation": True, "max_length": max_len }

    with model.trace("_"):
        temp_output = submodule.output.save()

    output_is_tuple = False

    if type(temp_output.shape) == tuple:
        output_is_tuple = True

    with model.trace(text, invoker_args=invoker_args):
        logits_original = model.output.save()
    logits_original = logits_original.value
    
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == 'in':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x = x * scale
        elif io == 'out':
            x = submodule.output
            if output_is_tuple: x = x[0]
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x = x * scale
        elif io == 'in_and_out':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x = x * scale
        else:
            raise ValueError(f"Invalid value for io: {io}")
        x = x.save()

    assert len(x.shape) == 3, f"Expected x to have shape (B, L, D), got {x.shape}, output_is_tuple: {output_is_tuple}"

    x_hat = dictionary(x).to(model.dtype)

    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == 'in':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x_hat = x_hat / scale
            submodule.input[:] = x_hat
        elif io == 'out':
            x = submodule.output
            if output_is_tuple: x = x[0]
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x_hat = x_hat / scale
            if output_is_tuple:
                submodule.output[0][:] = x_hat
            else:
                submodule.output[:] = x_hat
        elif io == 'in_and_out':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x_hat = x_hat / scale
            if output_is_tuple:
                submodule.output[0][:] = x_hat
            else:
                submodule.output[:] = x_hat
        else:
            raise ValueError(f"Invalid value for io: {io}")

        logits_reconstructed = model.output.save()
    logits_reconstructed = logits_reconstructed.value

    # logits when replacing component activations with zeros
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == 'in':
            x = submodule.input
            submodule.input[:] = t.zeros_like(x)
        elif io in ['out', 'in_and_out']:
            x = submodule.output
            if output_is_tuple:
                submodule.output[0][:] = t.zeros_like(x[0])
            else:
                submodule.output[:] = t.zeros_like(x)
        else:
            raise ValueError(f"Invalid value for io: {io}")
        
        input = model.inputs.save()
        logits_zero = model.output.save()

    logits_zero = logits_zero.value

    # get everything into the right format
    try:
        logits_original = logits_original.logits
        logits_reconstructed = logits_reconstructed.logits
        logits_zero = logits_zero.logits
    except:
        pass

    if isinstance(text, t.Tensor):
        tokens = text
    else:
        try:
            tokens = input[1]['input_ids']
        except:
            tokens = input[1]['input']

    # compute losses
    losses = []
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        loss_kwargs = {'ignore_index': model.tokenizer.pad_token_id}
    else:
        loss_kwargs = {}
    for logits in [logits_original, logits_reconstructed, logits_zero]:
        loss = t.nn.CrossEntropyLoss(**loss_kwargs)(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
        )
        losses.append(loss)

    return tuple(losses)


# Changed to context-level evaluation, other SAEs use average performance of all tokens in the context
@t.no_grad()
def evaluate(
    dictionary,  # dictionary model
    activations,  # activation source
    context_length=1024,  # context length
    batch_size=8,  # number of contexts per batch
    io="out",  # can be 'in' or 'out'
    normalize_batch=False,  # batch normalization switch
    tracer_args={'use_cache': False, 'output_attentions': False},
    device="cpu",
    n_batches: int = 1,
    is_dynamic_topk=False,  # whether it's a Dynamic TopK model
):
    assert n_batches > 0
    out = defaultdict(float)
    active_features = t.zeros(dictionary.dict_size, dtype=t.float32, device=device)

    # For tracking test samples
    data_tracking = {
        "unique_samples": set(),
        "sample_count": 0
    }
    
    print(f"Starting context-level evaluation: Model type={'Dynamic TopK' if is_dynamic_topk else 'Standard SAE'}")
    
    for batch_idx in range(n_batches):
        try:
            if isinstance(activations, (ActivationBuffer, NNsightActivationBuffer, DynamicTopKBuffer)):
                # Use existing buffer
                context_batch = activations.text_batch(batch_size=batch_size)

                # Track test samples
                for text in context_batch:
                    sample_id = text[:50] if len(text) > 50 else text
                    data_tracking["unique_samples"].add(sample_id)
                    data_tracking["sample_count"] += 1
                
                # Print data coverage information every 50 batches
                if batch_idx % 50 == 0 or batch_idx == n_batches - 1:
                    print(f"\nTest data statistics:")
                    print(f"  Processed {data_tracking['sample_count']} samples")
                    print(f"  Seen {len(data_tracking['unique_samples'])} different samples")
                    
                    # Print first 50 characters of the first 2 texts to check if data changes
                    print("\nChecking data batch samples:")
                    for i, text in enumerate(context_batch[:2]):
                        print(f"  Batch {batch_idx} sample {i}: {text[:50]}...")
                
                # Get context activations through the model
                with activations.model.trace(context_batch, **tracer_args, 
                              invoker_args={"truncation": True, "max_length": context_length}):
                    # Get activations from specified submodule
                    if io == "in":
                        context_acts = activations.submodule.inputs[0].save()
                    else:
                        context_acts = activations.submodule.output.save()
                    input_ids = activations.model.inputs.save()
                
                context_acts = context_acts.value
                if isinstance(context_acts, tuple):
                    context_acts = context_acts[0]
                
                # Get attention mask
                attn_mask = input_ids.value[1]["attention_mask"]
                
                if is_dynamic_topk:
                    last_token_indices = []
                    context_vectors = []
                    
                    for i in range(context_acts.shape[0]):
                        # Find position of last valid token in sequence
                        seq_len = attn_mask[i].sum().item()
                        if seq_len > 0:  # Ensure sequence is not empty
                            last_token_idx = seq_len - 1
                            last_token_indices.append(last_token_idx)
                            context_vectors.append(context_acts[i, last_token_idx])
                    
                    x = t.stack(context_vectors).to(device)
                    
                    pred_complexity = dictionary.probe(x)
                    # Map to k values
                    k_values = dictionary.probe.map_complexity_to_k(
                        pred_complexity,
                        min_k=dictionary.min_k,
                        max_k=dictionary.max_k,
                        base_k=dictionary.base_k
                    )
                    x_hat, f = dictionary(x, k_values=k_values, output_features=True)
                    out["avg_k"] += k_values.float().mean().item()
                    out["min_k"] += k_values.min().item()
                    out["max_k"] += k_values.max().item()
                    
                    l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean().item()
                    l1_loss = f.norm(p=1, dim=-1).mean().item() / t.norm(x, p=2, dim=-1).mean().item()
                    l0 = (f != 0).float().sum(dim=-1).mean().item()
                    
                    features_BF = t.flatten(f, start_dim=0, end_dim=-2).to(dtype=t.float32)
                    active_features += features_BF.sum(dim=0)
                    
                    # Cosine similarity
                    x_normed = x / t.linalg.norm(x, dim=-1, keepdim=True)
                    x_hat_normed = x_hat / t.linalg.norm(x_hat, dim=-1, keepdim=True)
                    cossim = (x_normed * x_hat_normed).sum(dim=-1).mean().item()
                    
                    # L2 ratio
                    l2_ratio = (t.linalg.norm(x_hat, dim=-1) / t.linalg.norm(x, dim=-1)).mean().item()
                    
                    # Fraction of variance explained
                    total_variance = t.var(x, dim=0).sum()
                    residual_variance = t.var(x - x_hat, dim=0).sum()
                    frac_variance_explained = (1 - residual_variance / total_variance).item()
                    
                    # Relative reconstruction bias
                    x_hat_norm_squared = t.linalg.norm(x_hat, dim=-1, ord=2)**2
                    x_dot_x_hat = (x * x_hat).sum(dim=-1)
                    relative_reconstruction_bias = (x_hat_norm_squared.mean() / x_dot_x_hat.mean()).item()
                    
                else:
                    # Baseline SAE: token-by-token evaluation
                    total_l2_loss = 0.0
                    total_l1_loss = 0.0 
                    total_l0 = 0.0
                    total_cossim = 0.0
                    total_l2_ratio = 0.0
                    total_frac_variance = 0.0
                    total_rel_recon_bias = 0.0
                    total_tokens = 0
                    batch_features = t.zeros(dictionary.dict_size, dtype=t.float32, device=device)
                    
                    # Process each context separately
                    for i in range(context_acts.shape[0]):
                        # Extract valid tokens
                        valid_mask = attn_mask[i] != 0
                        valid_tokens = context_acts[i, valid_mask]
                        
                        if len(valid_tokens) == 0:
                            continue

                        x = valid_tokens.to(device)
                        if normalize_batch:
                            x = x / x.norm(dim=-1).mean() * (dictionary.activation_dim ** 0.5)
                            
                        x_hat, f = dictionary(x, output_features=True)
                        
                        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
                        l1_loss = f.norm(p=1, dim=-1).mean() / t.norm(x, p=2, dim=-1).mean()
                        l0 = (f != 0).float().sum(dim=-1).mean()
                        
                        x_norm = t.linalg.norm(x, dim=-1, keepdim=True)
                        x_hat_norm = t.linalg.norm(x_hat, dim=-1, keepdim=True)
                        x_normed = x / x_norm
                        x_hat_normed = x_hat / x_hat_norm
                        cossim = (x_normed * x_hat_normed).sum(dim=-1).mean()
                        
                        # L2 ratio
                        l2_ratio = (t.linalg.norm(x_hat, dim=-1) / t.linalg.norm(x, dim=-1)).mean()
                        
                        # Fraction of variance explained
                        total_variance = t.var(x, dim=0).sum()
                        residual_variance = t.var(x - x_hat, dim=0).sum()
                        frac_variance_explained = (1 - residual_variance / total_variance)
                        
                        # Relative reconstruction bias
                        x_hat_norm_squared = t.linalg.norm(x_hat, dim=-1, ord=2)**2
                        x_dot_x_hat = (x * x_hat).sum(dim=-1)
                        relative_reconstruction_bias = x_hat_norm_squared.mean() / x_dot_x_hat.mean()
                        
                        # Feature usage
                        curr_features = t.flatten(f, start_dim=0, end_dim=-2).to(dtype=t.float32)
                        batch_features += curr_features.sum(dim=0)
                        
                        # Accumulate metrics
                        n_tokens = len(valid_tokens)
                        total_l2_loss += l2_loss.item() * n_tokens
                        total_l1_loss += l1_loss.item() * n_tokens
                        total_l0 += l0.item() * n_tokens
                        total_cossim += cossim.item() * n_tokens
                        total_l2_ratio += l2_ratio.item() * n_tokens
                        total_frac_variance += frac_variance_explained.item() * n_tokens
                        total_rel_recon_bias += relative_reconstruction_bias.item() * n_tokens
                        total_tokens += n_tokens
                    
                    # If there are valid tokens
                    if total_tokens > 0:
                        l2_loss = total_l2_loss / total_tokens
                        l1_loss = total_l1_loss / total_tokens
                        l0 = total_l0 / total_tokens
                        cossim = total_cossim / total_tokens
                        l2_ratio = total_l2_ratio / total_tokens
                        frac_variance_explained = total_frac_variance / total_tokens
                        relative_reconstruction_bias = total_rel_recon_bias / total_tokens
                        active_features += batch_features
                    else:
                        l2_loss = 0.0
                        l1_loss = 0.0
                        l0 = 0.0
                        cossim = 0.0
                        l2_ratio = 0.0
                        frac_variance_explained = 0.0
                        relative_reconstruction_bias = 0.0
                    
                    # Loss Recovered calculation for standard SAE
                    try:
                        # Use existing method to calculate loss_recovered
                        loss_original, loss_reconstructed, loss_zero = loss_recovered(
                            context_batch,
                            activations.model,
                            activations.submodule,
                            dictionary,
                            max_len=context_length,
                            normalize_batch=normalize_batch,
                            io=io,
                            tracer_args=tracer_args
                        )
                        
                        frac_recovered = (loss_reconstructed - loss_zero) / (loss_original - loss_zero)
                        
                        out["loss_original"] += loss_original.item()
                        out["loss_reconstructed"] += loss_reconstructed.item()
                        out["loss_zero"] += loss_zero.item()
                        out["frac_recovered"] += frac_recovered.item()
                        
                        print(f"Batch {batch_idx+1}/{n_batches}: "
                            f"L2 Loss = {l2_loss:.4f}, "
                            f"Frac. Recovered = {frac_recovered.item():.4f}")
                    except Exception as e:
                        print(f"Standard SAE Loss recovered calculation failed: {e}")
            
            out["l2_loss"] += l2_loss
            out["l1_loss"] += l1_loss
            out["l0"] += l0
            out["frac_variance_explained"] += frac_variance_explained
            out["cossim"] += cossim
            out["l2_ratio"] += l2_ratio
            out['relative_reconstruction_bias'] += relative_reconstruction_bias
            
            if is_dynamic_topk and 'avg_k' in out:
                print(f"Batch {batch_idx+1}/{n_batches}: "
                      f"Average k value = {out['avg_k']/(batch_idx+1):.2f}, "
                      f"Min k = {out['min_k']/(batch_idx+1):.2f}, "
                      f"Max k = {out['max_k']/(batch_idx+1):.2f}")

            print(f"Batch {batch_idx+1}/{n_batches}: "
                  f"L2 Loss = {l2_loss:.4f}, "
                  f"L0 = {l0:.2f}, "
                  f"Cosine similarity = {cossim:.4f}")
        
        except StopIteration:
            if batch_idx == 0:
                raise StopIteration("Insufficient data. Provide buffer with more samples or smaller batch size.")
            print(f"Data exhausted, completed only {batch_idx}/{n_batches} batches")
            break

    num_batches = batch_idx + 1
    out = {key: value / num_batches for key, value in out.items()}
    
    # Calculate active feature percentage
    frac_alive = (active_features != 0).float().sum() / dictionary.dict_size
    out["frac_alive"] = frac_alive.item()
    
    return out