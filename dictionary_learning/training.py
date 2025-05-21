import json
import torch.multiprocessing as mp
import os
from queue import Empty
from typing import Optional
from contextlib import nullcontext

import torch as t
from tqdm import tqdm

import wandb

from .dictionary import AutoEncoder
from .evaluation import evaluate
from .trainers.standard import StandardTrainer


def new_wandb_process(config, log_queue, entity, project):
    wandb.init(entity=entity, project=project, config=config, name=config["wandb_name"])
    while True:
        try:
            log = log_queue.get(timeout=1)
            if log == "DONE":
                break
            wandb.log(log)
        except Empty:
            continue
    wandb.finish()


def log_stats(
    trainers,
    step: int,
    act: t.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    log_queues: list=[],
    verbose: bool=False,
):
    with t.no_grad():
        # quick hack to make sure all trainers get the same x
        z = act.clone()
        for i, trainer in enumerate(trainers):
            log = {}
            act = z.clone()
            if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
                act = act[..., i, :]
            if not transcoder:
                act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                # fraction of variance explained
                total_variance = t.var(act, dim=0).sum()
                residual_variance = t.var(act - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                log[f"frac_variance_explained"] = frac_variance_explained.item()
            else:  # transcoder
                x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)

                # L0
                l0 = (f != 0).float().sum(dim=-1).mean().item()

            if verbose:
                print(f"Step {step}: L0 = {l0}, frac_variance_explained = {frac_variance_explained}")

            # log parameters from training
            log.update({f"{k}": v.cpu().item() if isinstance(v, t.Tensor) else v for k, v in losslog.items()})
            log[f"l0"] = l0
            trainer_log = trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                if isinstance(value, t.Tensor):
                    value = value.cpu().item()
                log[f"{name}"] = value

            if log_queues:
                log_queues[i].put(log)


def get_norm_factor(data, steps: int) -> float:
    total_mean_squared_norm = 0
    count = 0

    for step, act_BD in enumerate(tqdm(data, total=steps, desc="Calculating norm factor")):
        if step > steps:
            break

        count += 1
        mean_squared_norm = t.mean(t.sum(act_BD ** 2, dim=1))
        total_mean_squared_norm += mean_squared_norm

    average_mean_squared_norm = total_mean_squared_norm / count
    norm_factor = t.sqrt(average_mean_squared_norm).item()

    print(f"Average mean squared norm: {average_mean_squared_norm}")
    print(f"Norm factor: {norm_factor}")
    
    return norm_factor


def trainSAE(
    data,
    trainer_configs: list[dict],
    steps: int,
    use_wandb:bool=False,
    wandb_entity:str="",
    wandb_project:str="",
    save_steps:Optional[list[int]]=None,
    save_dir:Optional[str]=None,
    log_steps:Optional[int]=None,
    activations_split_by_head:bool=False,
    transcoder:bool=False,
    run_cfg:dict={},
    normalize_activations:bool=False,
    verbose:bool=False,
    device:str="cuda",
    autocast_dtype: t.dtype = t.float32,
    complexity_scores=None,
):
    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = nullcontext() if device_type == "cpu" else t.autocast(device_type=device_type, dtype=autocast_dtype)

    trainers = []
    for i, config in enumerate(trainer_configs):
        if "wandb_name" in config:
            config["wandb_name"] = f"{config['wandb_name']}_trainer_{i}"
        trainer_class = config["trainer"]
        del config["trainer"]

        if "scores_path" in config:
            del config["scores_path"]

        trainers.append(trainer_class(**config))

        dynamic_topk_steps = None
        for trainer in trainers:
            if hasattr(trainer, "__class__") and trainer.__class__.__name__ == "DynamicTopKTrainer":
                if hasattr(trainer, 'steps'):
                    dynamic_topk_steps = trainer.steps
                    print(f"Detected Dynamic TopK trainer, actual steps: {dynamic_topk_steps}")
                    break

        total_steps = dynamic_topk_steps if dynamic_topk_steps is not None else steps
        print(f"Total training steps set to: {total_steps}")
    
    original_complexity_scores = complexity_scores
    complexity_offset = 0 
    
    # After setting data and complexity scores, but don't call pre-training immediately
    if complexity_scores is not None:
        for trainer in trainers:
            if hasattr(trainer, "__class__") and trainer.__class__.__name__ == "DynamicTopKTrainer":
                print(f"Setting complete dataset and complexity scores for DynamicTopKTrainer")
                print(f"Complexity scores shape: {complexity_scores.shape}")
                
                trainer.data = data
                trainer.all_complexity_scores = complexity_scores

                if hasattr(trainer, 'config') and 'steps' in trainer.config:
                    actual_steps = trainer.config['steps']
                    print(f"Dynamic TopK actual training steps: {actual_steps}, instead of generic steps: {steps}")
                    dynamic_topk_steps = actual_steps
                else:
                    dynamic_topk_steps = steps

    print("Explicitly refreshing buffer to ensure data is available for training...")
    first_batch = next(iter(data))  # Trigger data buffer refresh

    # After ensuring buffer is refreshed, attempt pre-training
    for trainer in trainers:
        if hasattr(trainer, "__class__") and trainer.__class__.__name__ == "DynamicTopKTrainer":
            print("Now that buffer is ready, starting probe pre-training...")
            try:
                pretrain_result = trainer.pretrain_probe(trainer.data, trainer.all_complexity_scores)
                print(f"Pre-training completed, result: {pretrain_result}")
            except Exception as e:
                print(f"Pre-training failed: {e}")
                import traceback
                print(traceback.format_exc())

    wandb_processes = []
    log_queues = []

    if use_wandb:
        for i, trainer in enumerate(trainers):
            log_queue = mp.Queue()
            log_queues.append(log_queue)
            wandb_config = trainer.config | run_cfg
            # Make sure wandb config doesn't contain any CUDA tensors
            wandb_config = {k: v.cpu().item() if isinstance(v, t.Tensor) else v 
                          for k, v in wandb_config.items()}
            wandb_process = mp.Process(
                target=new_wandb_process,
                args=(wandb_config, log_queue, wandb_entity, wandb_project),
            )
            wandb_process.start()
            wandb_processes.append(wandb_process)

    if save_dir is not None:
        save_dirs = [
            os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
        ]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            # save config
            config = {"trainer": trainer.config}
            try:
                config["buffer"] = data.config
            except:
                pass
            with open(os.path.join(dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]

    if normalize_activations:
        norm_factor = get_norm_factor(data, steps=100)

        for trainer in trainers:
            trainer.config["norm_factor"] = norm_factor
            # Verify that all autoencoders have a scale_biases method
            trainer.ae.scale_biases(1.0)

    has_dynamic_topk = any(
        hasattr(trainer, "__class__") and trainer.__class__.__name__ == "DynamicTopKTrainer" 
        for trainer in trainers
    )

    if has_dynamic_topk:
        tqdm_total = dynamic_topk_steps
        print(f"Using Dynamic TopK specific steps as progress bar total: {tqdm_total}")
    else:
        tqdm_total = steps

    early_stop = False

    for step, act in enumerate(tqdm(data, total=total_steps)):
        act = act.to(dtype=autocast_dtype)

        if normalize_activations:
            act /= norm_factor

        if has_dynamic_topk:
            all_dynamic_completed = True
            for trainer in trainers:
                if hasattr(trainer, "__class__") and trainer.__class__.__name__ == "DynamicTopKTrainer":
                    if not hasattr(trainer, '_training_complete') or not trainer._training_complete:
                        all_dynamic_completed = False
                        break
            
            if all_dynamic_completed:
                print(f"All Dynamic TopK trainers have completed training, stopping early at step {step}/{tqdm_total}")
                early_stop = True
                break

        if step >= total_steps:
            print(f"Maximum steps {total_steps} reached, ending training")
            break

        # logging
        if (use_wandb or verbose) and step % log_steps == 0:
            log_stats(
                trainers, step, act, activations_split_by_head, transcoder, log_queues=log_queues, verbose=verbose
            )

        # saving
        if save_steps is not None and step in save_steps:
            for dir, trainer in zip(save_dirs, trainers):
                if dir is not None:
                    if normalize_activations:
                        # Temporarily scale up biases for checkpoint saving
                        trainer.ae.scale_biases(norm_factor)

                    if not os.path.exists(os.path.join(dir, "checkpoints")):
                        os.mkdir(os.path.join(dir, "checkpoints"))

                    checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                    t.save(
                        checkpoint,
                        os.path.join(dir, "checkpoints", f"ae_{step}.pt"),
                    )

                    if normalize_activations:
                        trainer.ae.scale_biases(1 / norm_factor)

        # training
        # for trainer in trainers:
        #     with autocast_context:
        #         trainer.update(step, act)
        for trainer in trainers:
            with autocast_context:
                if original_complexity_scores is not None and hasattr(trainer, "__class__") and trainer.__class__.__name__ == "DynamicTopKTrainer":
                    # Use offset to track complexity scores position
                    batch_size = act.shape[0]
                    if complexity_offset + batch_size <= len(original_complexity_scores):
                        batch_complexity = original_complexity_scores[complexity_offset:complexity_offset+batch_size]
                        complexity_offset += batch_size
                    else:
                        # Reset offset and reuse from beginning
                        print(f"Warning: Complexity scores exhausted (offset={complexity_offset}), starting over from beginning")
                        complexity_offset = 0
                        batch_size = min(batch_size, len(original_complexity_scores))
                        batch_complexity = original_complexity_scores[complexity_offset:complexity_offset+batch_size]
                        complexity_offset += batch_size
                    
                    trainer.update(step, act, complexity_scores=batch_complexity)
                else:
                    # Normal update for other trainers
                    trainer.update(step, act)
    
    if early_stop:
        print("Training terminated due to early completion by Dynamic TopK trainers")

    # save final SAEs
    for save_dir, trainer in zip(save_dirs, trainers):
        if normalize_activations:
            trainer.ae.scale_biases(norm_factor)
        if save_dir is not None:
            if hasattr(trainer, "__class__") and trainer.__class__.__name__ == "DynamicTopKTrainer":
                trainer.save_model(save_dir)
            else:
                final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                t.save(final, os.path.join(save_dir, "ae.pt"))

    # Signal wandb processes to finish
    if use_wandb:
        for queue in log_queues:
            queue.put("DONE")
        for process in wandb_processes:
            process.join()