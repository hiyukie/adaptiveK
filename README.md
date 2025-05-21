# AdaptiveK Sparse Autoencoders: Dynamic Sparsity Allocation for Interpretable LLM Representations

Adaptive Top K Sparse Autoencoders (AdaptiveK), a novel framework that dynamically adjusts sparsity levels based on the semantic complexity of each input. Experiments across three language models (Pythia-70M, Pythia-160M, and Gemma-2-2B) demonstrate that this complexity-driven adaptation significantly outperforms fixed-sparsity approaches on reconstruction fidelity, explained variance, and cosine similarity metrics while eliminating the computational burden of extensive hyperparameter tuning. 

AdaptiveK SAE is trained using the `dictionary_learning_demo` repo (`https://github.com/adamkarvonen/dictionary_learning_demo.git`), and it is evaluated using `SAEBench` (`https://github.com/adamkarvonen/SAEBench.git`).

# Usage

`python demo.py --save_dir ./dynamic_topk --model_name EleutherAI/pythia-160m-deduped --layers 3 --architectures dynamic_top_k --scores_path context_complexity_250k.parquet --test_data test_complexity.parquet --base_k 80 --min_k 20 --max_k 320 --l1_weight 0.005 --probe_weight 0.05 --sae_lr 1e-3 --probe_lr 1e-3 --phase_ratio 0.9`