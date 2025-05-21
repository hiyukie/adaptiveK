import types
import torch as t
from nnsight import LanguageModel
import argparse
import itertools
import os
import pandas as pd
import random
import json
import torch.multiprocessing as mp
import time
import huggingface_hub
from datasets import config
from demo_config import TrainerType

import demo_config
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.evaluation import evaluate
from dictionary_learning.training import trainSAE
import dictionary_learning.utils as utils
from dictionary_learning.dynamic_buffer import DynamicTopKBuffer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="where to store sweep")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb logging")
    parser.add_argument("--dry_run", action="store_true", help="dry run sweep")
    parser.add_argument("--save_checkpoints", action="store_true", help="save checkpoints")
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True, help="layers to train SAE on"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="which language model to use",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        choices=[e.value for e in demo_config.TrainerType],
        required=True,
        help="which SAE architectures to train",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device to train on")
    parser.add_argument("--hf_repo_id", type=str, help="Hugging Face repo ID to push results to")

    # Additional parameters for dynamic TopK
    parser.add_argument("--base_k", type=int, default=None, help="Base k value for Dynamic TopK")
    parser.add_argument("--min_k", type=int, default=None, help="Minimum k value for Dynamic TopK")
    parser.add_argument("--max_k", type=int, default=None, help="Maximum k value for Dynamic TopK")
    parser.add_argument("--l1_weight", type=float, default=None, help="L1 weight for Dynamic TopK")
    parser.add_argument("--probe_weight", type=float, default=None, help="Probe weight for Dynamic TopK")
    parser.add_argument("--auxk_alpha", type=float, default=None, help="Auxiliary loss weight (1/32)")
    parser.add_argument("--sae_lr", type=float, default=None, help="SAE learning rate")
    parser.add_argument("--probe_lr", type=float, default=None, help="Probe learning rate")
    parser.add_argument("--phase_ratio", type=float, default=None, help="Ratio of steps for SAE pretraining (0-1)")
    parser.add_argument("--scores_path", type=str, default="pythia_complexity.parquet",
                        help="Path to precomputed complexity scores")

    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to test data file with complexity scores")

    args = parser.parse_args()
    return args


def run_sae_training(
    model_name: str,
    layer: int,
    save_dir: str,
    device: str,
    architectures: list,
    num_tokens: int,
    random_seeds: list[int],
    dictionary_widths: list[int],
    learning_rates: list[float],
    args: argparse.Namespace,
    dry_run: bool = False,
    use_wandb: bool = False,
    save_checkpoints: bool = False,
    # For dynamic SAE
    # buffer_tokens: int = 50_000_000,
    buffer_tokens: int = 256_000_000,
    # For baseline SAEs
    # buffer_tokens: int = 1250,
):
    
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

    # model and data parameters
    context_length = demo_config.LLM_CONFIG[model_name].context_length

    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    sae_batch_size = demo_config.LLM_CONFIG[model_name].sae_batch_size
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    num_buffer_inputs = buffer_tokens // context_length
    print(f"buffer_size: {num_buffer_inputs}, buffer_size_in_tokens: {buffer_tokens}")

    log_steps = 100  # Log the training on wandb or print to console every log_steps

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train

    complexity_scores = None
    if TrainerType.DYNAMIC_TOP_K.value in architectures and args.scores_path:
        try:
            import pandas as pd
            print(f"Loading complexity scores: {args.scores_path}")
            complexity_df = pd.read_parquet(args.scores_path)
            print(f"Successfully loaded complexity scores, containing {len(complexity_df)} records")
            
            # Convert scores to tensor and move to device
            complexity_scores = t.tensor(complexity_df['complexity_score'].values, device=device, dtype=t.float32)
        except Exception as e:
            print(f"Error loading complexity scores: {e}")

    if save_checkpoints:
        # Creates checkpoints at 0.0%, 0.1%, 0.316%, 1%, 3.16%, 10%, 31.6%, 100% of training
        desired_checkpoints = t.logspace(-3, 0, 7).tolist()
        desired_checkpoints = [0.0] + desired_checkpoints[:-1]
        desired_checkpoints.sort()
        print(f"desired_checkpoints: {desired_checkpoints}")

        save_steps = [int(steps * step) for step in desired_checkpoints]
        save_steps.sort()
        print(f"save_steps: {save_steps}")
    else:
        save_steps = None

    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)
    submodule = utils.get_submodule(model, layer)
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    activation_dim = model.config.hidden_size

    # generator = hf_dataset_to_generator("monology/pile-uncopyrighted")

    if TrainerType.DYNAMIC_TOP_K.value in architectures and args.scores_path:
        import pandas as pd
        from tqdm import tqdm
        print(f"Creating custom generator for Dynamic TopK SAE, using parquet file: {args.scores_path}")
        
        df = pd.read_parquet(args.scores_path)
        texts = df['text'].tolist()

        tokenizer = model.tokenizer

        # print("\nAnalyzing token length for first 10 samples:")
        # for i, text in enumerate(texts[:10]):
        #     # Tokenize the text
        #     tokens = tokenizer(text, return_tensors="pt")
        #     
        #     # Get token count (excluding padding)
        #     token_count = tokens['input_ids'].shape[1]
        #     
        #     # Print token count and complexity score
        #     print(f"Sample #{i}: tokens={token_count}, complexity score={df['complexity_score'].iloc[i]:.2f}")
        
        complexity_scores = t.tensor(df['complexity_score'].values, device=device, dtype=t.float32)
        print(f"complexity score: min={complexity_scores.min().item():.4f}, max={complexity_scores.max().item():.4f}, mean={complexity_scores.mean().item():.4f}")
        
        def generator_factory():
            print("Creating new text generator...")
            
            def generator():
                for text in texts:
                    yield text
                    
            return generator()
        
        generator = generator_factory()
        
        print("Created new text generator from Parquet file: context_complexity.parquet")
        
        activation_buffer = DynamicTopKBuffer(
            generator,
            model,
            submodule,
            n_ctxs=num_buffer_inputs,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            out_batch_size=sae_batch_size,
            io=io,
            d_submodule=activation_dim,
            device=device,
            data_factory=generator_factory
        )
    else:
        generator = hf_dataset_to_generator("monology/pile-uncopyrighted")
        
        activation_buffer = ActivationBuffer(
            generator,
            model,
            submodule,
            n_ctxs=num_buffer_inputs,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            out_batch_size=sae_batch_size,
            io=io,
            d_submodule=activation_dim,
            device=device,
        )

    # activation_buffer = ActivationBuffer(
    #     generator,
    #     model,
    #     submodule,
    #     n_ctxs=num_buffer_inputs,
    #     ctx_len=context_length,
    #     refresh_batch_size=llm_batch_size,
    #     out_batch_size=sae_batch_size,
    #     io=io,
    #     d_submodule=activation_dim,
    #     device=device,
    # )

    trainer_configs = demo_config.get_trainer_configs(
        architectures,
        learning_rates,
        random_seeds,
        activation_dim,
        dictionary_widths,
        model_name,
        device,
        layer,
        submodule_name,
        steps,
    )

    if TrainerType.DYNAMIC_TOP_K.value in architectures:
        for i, config in enumerate(trainer_configs):
            if config.get('trainer_class') == 'DynamicTopKTrainer':
                if args.base_k is not None:
                    trainer_configs[i]['base_k'] = args.base_k
                if args.min_k is not None:
                    trainer_configs[i]['min_k'] = args.min_k
                if args.max_k is not None:
                    trainer_configs[i]['max_k'] = args.max_k
                if args.l1_weight is not None:
                    trainer_configs[i]['l1_weight'] = args.l1_weight
                if args.probe_weight is not None:
                    trainer_configs[i]['probe_weight'] = args.probe_weight
                if args.auxk_alpha is not None:
                    trainer_configs[i]['auxk_alpha'] = args.auxk_alpha
                if args.sae_lr is not None:
                    trainer_configs[i]['sae_lr'] = args.sae_lr
                if args.probe_lr is not None:
                    trainer_configs[i]['probe_lr'] = args.probe_lr
                if args.phase_ratio is not None:
                    trainer_configs[i]['phase_ratio'] = args.phase_ratio
                if 'scores_path' in trainer_configs[i]:
                    del trainer_configs[i]['scores_path']

    print(f"len trainer configs: {len(trainer_configs)}")
    assert len(trainer_configs) > 0
    save_dir = f"{save_dir}/{submodule_name}"

    if not dry_run:
        extra_args = {}
        if complexity_scores is not None:
            print(f"Complexity scores shape: {complexity_scores.shape}, total samples: {len(complexity_scores)}")

        # Pass complete data and scores for DynamicTopK
        extra_args = {}
        if complexity_scores is not None:
            extra_args['complexity_scores'] = complexity_scores
            # Iterator for data reuse
            # extra_args['data_iterator'] = activation_buffer

        # actually run the sweep
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            use_wandb=use_wandb,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
            wandb_project=demo_config.wandb_project,
            normalize_activations=False,
            verbose=False,
            autocast_dtype=t.bfloat16,
            **extra_args,
        )


@t.no_grad()
def eval_saes(
    model_name: str,
    ae_paths: list[str],
    n_inputs: int,
    device: str,
    overwrite_prev_results: bool = False,
    transcoder: bool = False,
    test_data_path: str = None,  # Test data path
) -> dict:
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

    if transcoder:
        io = "in_and_out"
    else:
        io = "out"

    context_length = demo_config.LLM_CONFIG[model_name].context_length
    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    loss_recovered_batch_size = max(llm_batch_size // 5, 1)
    sae_batch_size = loss_recovered_batch_size * context_length
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)

    # Explicitly load test data
    print(f"Using specified test data: {test_data_path}")
    import pandas as pd
    df = pd.read_parquet(test_data_path)
    test_texts = df['text'].tolist()
    print(f"Loaded {len(test_texts)} test samples")
    
    # Total test samples
    buffer_size = n_inputs
    io = "out"
    n_batches = buffer_size // loss_recovered_batch_size
    
    print(f"Setting evaluation batch count to {n_batches}, with {loss_recovered_batch_size} samples per batch")
    print(f"Will evaluate a total of {buffer_size} samples")

    eval_results = {}

    for ae_path in ae_paths:
        output_filename = f"{ae_path}/eval_results.json"
        complexity_output_dir = f"{ae_path}/complexity_evaluation"
        if not overwrite_prev_results:
            if os.path.exists(output_filename):
                print(f"Skipping {ae_path} because evaluation results already exist")
                continue

        dictionary, config = utils.load_dictionary(ae_path, device)
        dictionary = dictionary.to(dtype=model.dtype)

        layer = config["trainer"]["layer"]
        submodule = utils.get_submodule(model, layer)

        activation_dim = config["trainer"]["activation_dim"]

        is_dynamic_topk = (
            "dict_class" in config["trainer"] and 
            config["trainer"]["dict_class"] == "AutoEncoderDynamicTopK"
        )

        # Create new buffer, ensuring use of test data
        def test_generator_factory():
            print("Creating new test text generator...")
            return iter(test_texts)
        
        # For all model types, use fresh buffer instances
        if is_dynamic_topk:
            print(f"Creating evaluation buffer for Dynamic TopK...")
            activation_buffer = DynamicTopKBuffer(
                test_generator_factory(),
                model,
                submodule,
                n_ctxs=buffer_size,
                ctx_len=context_length,
                refresh_batch_size=llm_batch_size,
                out_batch_size=sae_batch_size, 
                io=io,
                d_submodule=activation_dim,
                device=device,
                data_factory=test_generator_factory
            )
            activation_buffer._evaluation_mode = True
            # Ensure buffer doesn't reuse cached data
            activation_buffer.cached_complete = False
        else:
            print(f"Creating evaluation buffer for standard SAE...")
            activation_buffer = ActivationBuffer(
                test_generator_factory(),
                model,
                submodule,
                n_ctxs=buffer_size,
                ctx_len=context_length,
                refresh_batch_size=llm_batch_size,
                out_batch_size=sae_batch_size,
                io=io,
                d_submodule=activation_dim,
                device=device,
            )

        print(f"Starting standard evaluation for model: {ae_path}")
        eval_results = evaluate(
            dictionary,
            activation_buffer,
            context_length=context_length,
            batch_size=loss_recovered_batch_size,
            io=io,
            device=device,
            n_batches=n_batches,
            is_dynamic_topk=is_dynamic_topk
        )

        hyperparameters = {
            "n_inputs": buffer_size,
            "context_length": context_length,
        }
        eval_results["hyperparameters"] = hyperparameters

        print(eval_results)

        with open(output_filename, "w") as f:
            json.dump(eval_results, f)

    return eval_results


def push_to_huggingface(save_dir: str, repo_id: str):
    api = huggingface_hub.HfApi()

    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=save_dir,
    )


if __name__ == "__main__":
    """python demo.py --save_dir ./run2 --model_name EleutherAI/pythia-70m-deduped --layers 3 --architectures standard jump_relu batch_top_k top_k gated --use_wandb
    python demo.py --save_dir ./run3 --model_name google/gemma-2-2b --layers 12 --architectures standard top_k --use_wandb
    python demo.py --save_dir ./jumprelu --model_name EleutherAI/pythia-70m-deduped --layers 3 --architectures jump_relu --use_wandb"""
    args = get_args()

    hf_repo_id = args.hf_repo_id

    if hf_repo_id:
        assert huggingface_hub.repo_exists(repo_id=hf_repo_id, repo_type="model")

    # This prevents random CUDA out of memory errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    mp.set_start_method("spawn", force=True)

    config.STREAMING_READ_MAX_RETRIES = 100
    config.STREAMING_READ_RETRY_INTERVAL = 20

    start_time = time.time()

    save_dir = f"{args.save_dir}_{args.model_name}_{'_'.join(args.architectures)}".replace("/", "_")

    for layer in args.layers:
        run_sae_training(
            model_name=args.model_name,
            layer=layer,
            save_dir=save_dir,
            device=args.device,
            architectures=args.architectures,
            num_tokens=demo_config.num_tokens,
            random_seeds=demo_config.random_seeds,
            dictionary_widths=demo_config.dictionary_widths,
            learning_rates=demo_config.learning_rates,
            args=args,
            dry_run=args.dry_run,
            use_wandb=args.use_wandb,
            save_checkpoints=args.save_checkpoints,
        )

    ae_paths = utils.get_nested_folders(save_dir)

    eval_saes(
        args.model_name,
        ae_paths,
        demo_config.eval_num_inputs,
        args.device,
        overwrite_prev_results=True,
        test_data_path=args.test_data  # Pass test data
    )

    print(f"Total time: {time.time() - start_time}")

    if hf_repo_id:
        push_to_huggingface(save_dir, hf_repo_id)