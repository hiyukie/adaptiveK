import os
import json
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Any
import pandas as pd

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)

class LinearProbe(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 512,
        lambda_reg: float = 1.0,
        min_score: float = 0.0,
        max_score: float = 10.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.lambda_reg = lambda_reg
        self.min_score = min_score
        self.max_score = max_score

        self.weights = nn.Parameter(t.zeros(input_dim, 1))
        self.bias = nn.Parameter(t.zeros(1))
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.training and self.lambda_reg > 0:
            # L2
            reg_loss = self.lambda_reg * t.sum(self.weights ** 2) / 2
            self.reg_loss = reg_loss
            
        # y = xw + b
        raw_scores = t.matmul(x, self.weights) + self.bias
        scores = t.clamp(raw_scores, self.min_score, self.max_score)
        
        return scores.squeeze(-1)
    
    def pretrain(self, X, y, val_X=None, val_y=None):
        device = X.device
        n_samples = X.shape[0]
        
        X_float32 = X.to(t.float32)
        y_float32 = y.to(t.float32)
        
        X_with_bias = t.cat([X_float32, t.ones(n_samples, 1, device=device, dtype=t.float32)], dim=1)
        y_float32 = y_float32.view(-1, 1)
        
        reg_matrix = t.eye(X_with_bias.shape[1], device=device, dtype=t.float32)
        reg_matrix[-1, -1] = 0 
        
        XTX = t.matmul(X_with_bias.t(), X_with_bias)
        A = XTX + self.lambda_reg * reg_matrix 
        b = t.matmul(X_with_bias.t(), y_float32)  
        
        A = A.to(t.float32)
        b = b.to(t.float32)
        
        solution = t.linalg.solve(A, b)
        
        with t.no_grad():
            self.weights.copy_(solution[:-1])
            self.bias.copy_(solution[-1])
        
        return self._compute_metrics(X, y, val_X, val_y)
    
    def _compute_metrics(self, X, y, val_X=None, val_y=None):

        metrics = {}
        
        train_preds = self.forward(X)
        train_mse = F.mse_loss(train_preds, y.squeeze()).item()
        train_rmse = np.sqrt(train_mse)
        train_r2 = self._compute_r2(train_preds, y.squeeze())
        
        metrics.update({
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
        })
        
        # Validation metrics
        if val_X is not None and val_y is not None:
            val_preds = self.forward(val_X)
            val_mse = F.mse_loss(val_preds, val_y).item()
            val_rmse = np.sqrt(val_mse)
            val_r2 = self._compute_r2(val_preds, val_y)
            
            metrics.update({
                'val_mse': val_mse,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
            })
        
        return metrics
    
    def _compute_r2(self, preds, targets):

        total_variance = t.var(targets) * len(targets)
        residual_variance = t.sum((targets - preds) ** 2)
        
        r2 = 1.0 - (residual_variance / total_variance)
        return r2.item()
    
    def cross_validate(self, X, y, lambda_values, n_folds=5):

        device = X.device
        n_samples = X.shape[0]

        X_float32 = X.to(t.float32)
        y_float32 = y.to(t.float32)
        
        fold_size = n_samples // n_folds
        indices = t.randperm(n_samples)
        
        lambda_metrics = {}
        
        for lambda_val in lambda_values:
            fold_metrics = []
            
            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples
                val_indices = indices[val_start:val_end]
                train_indices = t.cat([indices[:val_start], indices[val_end:]])
                
                fold_train_X = X[train_indices]
                fold_train_y = y[train_indices]
                fold_val_X = X[val_indices]
                fold_val_y = y[val_indices]
                
                probe = LinearProbe(
                    input_dim=self.input_dim,
                    lambda_reg=lambda_val,
                    min_score=self.min_score,
                    max_score=self.max_score,
                ).to(device)
                
                # Train probe
                metrics = probe.pretrain(fold_train_X, fold_train_y, fold_val_X, fold_val_y)
                fold_metrics.append(metrics['val_rmse'])
            
            # average RMSE for this lambda
            avg_rmse = sum(fold_metrics) / len(fold_metrics)
            lambda_metrics[lambda_val] = {'avg_val_rmse': avg_rmse, 'fold_rmses': fold_metrics}
            
        # best lambda
        best_lambda = min(lambda_metrics.keys(), key=lambda l: lambda_metrics[l]['avg_val_rmse'])
        self.lambda_reg = best_lambda
        
        return best_lambda, lambda_metrics

    def compute_loss(self, x, targets):
        
        predictions = self.forward(x)
        mse_loss = F.mse_loss(predictions, targets)
        l2_reg = self.lambda_reg * t.sum(self.weights ** 2) / 2

        total_loss = mse_loss + l2_reg

        return total_loss
    
    def map_complexity_to_k(self, complexity_scores, min_k=20, max_k=320, base_k=80):

        normalized_scores = (complexity_scores - self.min_score) / (self.max_score - self.min_score)
    
        mid_point = 0.5  # sigmoid midpoint, corresponds to base_k
        steepness = 6.0  # controls curve steepness
        
        sigmoid_factor = 1.0 / (1.0 + t.exp(-steepness * (normalized_scores - mid_point)))
        k_values = min_k + sigmoid_factor * (max_k - min_k)
        
        return t.clamp(k_values, min_k, max_k).round().int()


class AutoEncoderDynamicTopK(Dictionary, nn.Module):

    def __init__(self, activation_dim, dict_size, base_k=80, min_k=20, max_k=320):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.base_k = base_k
        self.min_k = min_k
        self.max_k = max_k

        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.b_dec = nn.Parameter(t.zeros(activation_dim))
        
        self._init_weights()
    
    def _init_weights(self):

        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, self.activation_dim, self.dict_size
        )

        self.encoder.weight.data = self.decoder.weight.T.clone()
        # Initialize encoder bias to zero
        self.encoder.bias.data.zero_()

    def encode(self, x, k_values=None):

        # encoder
        pre_activations = self.encoder(x - self.b_dec)
        post_relu_acts = F.relu(pre_activations)
        
        if k_values is None:
            k_values = t.full((x.shape[0],), self.base_k, device=x.device, dtype=t.int)
        
        # k largest activation values are chosen to be 0
        result = t.zeros_like(post_relu_acts)
        for i, (sample, k) in enumerate(zip(post_relu_acts, k_values)):
            k_int = int(k.item())
            values, indices = t.topk(sample, k=min(k_int, self.dict_size), dim=-1)
            result[i].scatter_(-1, indices, values)
        return result, {"k_values": k_values, "post_relu_acts": post_relu_acts}

    def decode(self, x):
        return self.decoder(x) + self.b_dec

    def forward(self, x, k_values=None, output_features=False):

        with t.no_grad():
            pred_complexity = self.probe(x)
            k_values = self.probe.map_complexity_to_k(
                pred_complexity,
                min_k=self.min_k,
                max_k=self.max_k,
                base_k=self.base_k
            )

        encoded_acts, info = self.encode(x, k_values)
        x_hat = self.decode(encoded_acts)
        
        if not output_features:
            return x_hat
        else:
            return x_hat, encoded_acts

    def scale_biases(self, scale):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale

    @classmethod
    def from_pretrained(cls, path, base_k=80, min_k=20, max_k=320, device=None):

        print(f"load pretrained model, from: {path}")
        try:
            # ae
            ae_path = path if path.endswith(".pt") else os.path.join(path, "ae.pt")
            print(f"load SAE: {ae_path}")
            
            state_dict = t.load(ae_path)
            dict_size, activation_dim = state_dict["encoder.weight"].shape
            print(f"SAE dict_size: {dict_size}, activation_dim: {activation_dim}")

            autoencoder = cls(activation_dim, dict_size, base_k, min_k, max_k)
            autoencoder.load_state_dict(state_dict)
            
            # load probe
            probe_dir = os.path.dirname(ae_path)
            probe_path = os.path.join(probe_dir, "probe.pt")
            print(f"probe: {probe_path}")

            if device is not None:
                autoencoder.to(device)
                if hasattr(autoencoder, 'probe'):
                    autoencoder.probe.to(device)
            
            print("Loading model complete")
            return autoencoder
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            import traceback
            print(traceback.format_exc())
            raise


class DynamicTopKTrainer(SAETrainer):

    def __init__(
        self,
        steps,
        activation_dim,
        dict_size,
        base_k,
        min_k,
        max_k,
        layer,
        lm_name,
        sae_lr,
        probe_lr,
        l1_weight,
        probe_weight,
        dict_class,
        auxk_alpha=1/32,
        phase_ratio=0.9,
        device="cuda",
        wandb_name="DynamicTopKSAE",
        seed=None,
        submodule_name=None,
        warmup_steps=None,
        decay_start=None,
        data=None,
        all_complexity_scores=None, 
    ):
        super().__init__(seed)

        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name
        self.steps = steps
        self.warmup_steps = warmup_steps
        self.decay_start = decay_start
        self.phase_ratio = phase_ratio
        self.dict_class = dict_class
        
        self.base_k = base_k
        self.min_k = min_k
        self.max_k = max_k
        self.l1_weight = l1_weight
        self.probe_weight = probe_weight
        self.auxk_alpha = auxk_alpha
        
        self.sae_lr = sae_lr
        self.probe_lr = probe_lr
        
        self.device = device if device is not None else ("cuda" if t.cuda.is_available() else "cpu")

        self.data = data
        self.all_complexity_scores = all_complexity_scores

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        self.ae = AutoEncoderDynamicTopK(activation_dim, dict_size, base_k, min_k, max_k)
        self.probe = LinearProbe(input_dim=activation_dim)
        
        self.ae.to(self.device)
        self.probe.to(self.device)
        
        self.current_phase = 0  # 0:not started, 1:probe pretrain, 2:SAE pretrain, 3:joint finetune
        
        self.sae_optimizer = t.optim.Adam(self.ae.parameters(), lr=sae_lr)
        self.probe_optimizer = t.optim.Adam(self.probe.parameters(), lr=probe_lr)
        
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)
        
        self.sae_scheduler = t.optim.lr_scheduler.LambdaLR(self.sae_optimizer, lr_lambda=lr_fn)
        self.probe_scheduler = t.optim.lr_scheduler.LambdaLR(self.probe_optimizer, lr_lambda=lr_fn)
        
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        self.dead_feature_threshold = 10_000
        self.top_k_aux = activation_dim // 2
        
        self.logging_parameters = [
            "effective_l0", 
            "dead_features", 
            "probe_loss", 
            "recon_loss", 
            "sparsity_loss",
            "probe_rmse",
            "pre_norm_auxk_loss"
        ]
        self.effective_l0 = -1
        self.dead_features = -1
        self.probe_loss = -1
        self.recon_loss = -1
        self.sparsity_loss = -1
        self.probe_rmse = -1
        self.pre_norm_auxk_loss = -1
        
        self.initial_probe_weights = None
        self.initial_probe_bias = None
        
        self.probe_loss_history = []
        self.cur_deviation_weight = 0.2 
        self.loss_stability_threshold = 0.05
        
    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):

        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.top_k_aux, self.dead_features)
            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)

            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(dim=-1, index=auxk_indices, src=auxk_acts)

            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(residual_BD.shape)
            loss_denom = (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)
    

    def pretrain_probe(self, x, complexity_scores, lambda_values=None):

        lambda_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        
        print("\n" + "="*50)
        print(f"PHASE 1: Pretraining Linear Probe using Ridge Regression")
        print("="*50)

        if hasattr(self, '_pretrain_completed') and self._pretrain_completed:
            print("pretraining is completed, skip it....")

        if not isinstance(x, t.Tensor):
            print(f"complexity score shape: {complexity_scores.shape}")
            
            print("collecting training contexts...")
            all_activations = []
            all_scores = []
            sample_counter = 0
            batch_counter = 0
            
            from tqdm import tqdm
            pbar = tqdm(desc="collecting...")
            
            try:
                for batch in x:
                    batch_counter += 1
                    if not isinstance(batch, t.Tensor):
                        print(f"batch #{batch_counter} type {type(batch)}")
                        continue
                        
                    batch_size = batch.shape[0]
                    print(f"batch #{batch_counter}: shape={batch.shape}, device={batch.device}, type={batch.dtype}")
                    print(f"batch #{batch_counter}, min={batch.min().item():.4f}, max={batch.max().item():.4f}, mean={batch.mean().item():.4f}")
                    
                    if sample_counter + batch_size <= len(complexity_scores):
                        batch_scores = complexity_scores[sample_counter:sample_counter+batch_size]
                        all_activations.append(batch.to(self.device))
                        all_scores.append(batch_scores.to(self.device))
                        
                        old_counter = sample_counter
                        sample_counter += batch_size
                        pbar.update(batch_size)
                        
                        print(f"already collected: {old_counter}->{sample_counter}/{len(complexity_scores)} (+{batch_size})")
                        
                    else:
                        valid_size = len(complexity_scores) - sample_counter
                        if valid_size > 0:
                            batch_scores = complexity_scores[sample_counter:sample_counter+valid_size]
                            all_activations.append(batch[:valid_size].to(self.device))
                            all_scores.append(batch_scores.to(self.device))
                            
                            old_counter = sample_counter
                            sample_counter += valid_size
                            pbar.update(valid_size)
                            
                            print(f"have collected{valid_size}samples: {old_counter}->{sample_counter}/{len(complexity_scores)}")
                        break
                    
                    if sample_counter >= 250000:
                        print(f"already collected all ({sample_counter})")
                        break
                        
            except Exception as e:
                print(f"error when collecting contexts: {e}")
                import traceback
                print(traceback.format_exc())
            finally:
                pbar.close()
            
            print(f"number of batches: {batch_counter}, number of samples: {sample_counter}")
            
            all_x = t.cat(all_activations, dim=0)
            all_y = t.cat(all_scores, dim=0)

            print(f"all_x shape={all_x.shape}, all_y shape={all_y.shape}")

            del all_activations, all_scores
            t.cuda.empty_cache()
            
            print(f"successfully collected {all_x.shape[0]} samples for training")
            print(f"activation shape: {all_x.shape}")
            print(f"complexity score shape: {all_y.shape}")
            
            min_score = all_y.min().item()
            max_score = all_y.max().item()
            mean_score = all_y.mean().item()
            std_score = all_y.std().item()
            
            print(f"complexity score:")
            print(f"  Min: {min_score:.4f}, Max: {max_score:.4f}")
            print(f"  Mean: {mean_score:.4f}, Std: {std_score:.4f}")
            
            # training set, validation set
            n_total = all_x.shape[0]
            n_train = int(n_total * 0.8)
            indices = t.randperm(n_total, device=self.device)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            
            train_x = all_x[train_indices]
            train_y = all_y[train_indices]
            val_x = all_x[val_indices]
            val_y = all_y[val_indices]
            
            del all_x, all_y, indices
            t.cuda.empty_cache()
            
            print("to find best lambda...")
            best_lambda, lambda_metrics = self.probe.cross_validate(
                train_x, train_y, lambda_values
            )
            
            print(f"best lambda: {best_lambda}")
            for lambda_val, metrics in lambda_metrics.items():
                print(f"Lambda {lambda_val}: Avg RMSE = {metrics['avg_val_rmse']:.4f}")
        else:
            raise ValueError("error")
        
        print("use best lambda on all dataset to train linear probe...")
        self.probe.lambda_reg = best_lambda

        self._pretrain_completed = True
        metrics = self.probe.pretrain(train_x, train_y)
        print("linear probe training is completed.")

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        
        with t.no_grad():
            sample_size = len(train_x)
            test_x = train_x[:sample_size]
            test_y = train_y[:sample_size]
            
            train_preds = self.probe(test_x)
            
            print("\n complexity score:")
            print(f"  true - min: {test_y.min().item():.2f}, max: {test_y.max().item():.2f}, mean: {test_y.mean().item():.2f}")
            print(f"  predit - min: {train_preds.min().item():.2f}, max: {train_preds.max().item():.2f}, mean: {train_preds.mean().item():.2f}")
            
            from scipy.stats import pearsonr
            true_np = test_y.cpu().numpy()
            pred_np = train_preds.cpu().numpy()
            correlation, _ = pearsonr(true_np, pred_np)
            print(f"  Pearson's correlation coefficient: {correlation:.4f}")

            k_values = self.probe.map_complexity_to_k(
                train_preds,
                min_k=self.min_k,
                max_k=self.max_k,
                base_k=self.base_k
            )
            import numpy as np
            k_np = k_values.cpu().numpy()
            print(f"\n predit k range:")
            print(f"  Min: {k_np.min()}, Max: {k_np.max()}, Mean: {k_np.mean():.2f}")

            test_batch = next(iter(self.data))
            test_batch_size = test_batch.shape[0]
            
            batch_scores = self.all_complexity_scores[:test_batch_size]
            
            batch_pred = self.probe(test_batch)
            batch_loss = self.probe.compute_loss(test_batch, batch_scores)

        self.initial_probe_weights = self.probe.weights.data.clone()
        self.initial_probe_bias = self.probe.bias.data.clone()
        
        self.metrics = metrics
        
        return metrics

    def update(self, step, x, complexity_scores=None):

        if not hasattr(self, '_actual_step'):
            self._actual_step = 0
    
        if self._actual_step >= self.steps:
            if not hasattr(self, '_reported_max_steps'):
                print(f"\n dynamic topk already reached max stpes {self.steps}")
                self._reported_max_steps = True
                self._training_complete = True
            return 0.0

        self._actual_step += 1
        
        if self._actual_step % 100 == 0 or self._actual_step == 1:
            remaining = self.steps - self._actual_step
            print(f"\n dynamic topk actual training step: {self._actual_step}/{self.steps}, remain: {remaining}, phase: {self.current_phase}")
        
        # Phase 1: Pretrain probe
        if step == 0:
            print("\n" + "="*50)
            print(f"PHASE 1 is done in pretrained_probe")
            print("="*50)

            train_mse = self.metrics.get('train_mse', 0)
            train_rmse = self.metrics.get('train_rmse', 0)
            print(f"pretrain - Loss: {train_mse:.6f}, RMSE: {train_rmse:.6f}")
            loss = train_mse

            self.probe_loss = loss
            self.probe_rmse = t.sqrt(t.tensor(loss)).item()
            
            return loss
            
        # Phase 2: SAE pretraining (frozen probe)
        elif step < int(self.steps * self.phase_ratio):
            if step == 1:
                print("\n" + "="*50)
                print(f"PHASE 2: Pretrain SAE with Frozen Probe")
                print("="*50)

            self.current_phase = 2
            # Freeze probe parameters
            for param in self.probe.parameters():
                param.requires_grad = False
                
            # Get complexity scores and k values
            with t.no_grad():
                pred_complexity = self.probe(x)
                k_values = self.probe.map_complexity_to_k(
                    pred_complexity,
                    min_k=self.min_k,
                    max_k=self.max_k,
                    base_k=self.base_k
                )
            
            if step == 1:
                from ..trainers.top_k import geometric_median
                self.ae.b_dec.data = geometric_median(x)
            
            # SAE forward
            post_relu_acts = F.relu(self.ae.encoder(x - self.ae.b_dec))
            encoded_acts, info = self.ae.encode(x, k_values)
            x_hat = self.ae.decode(encoded_acts)
            
            residual = x - x_hat

            self._update_feature_usage(encoded_acts)
            
            recon_loss = F.mse_loss(x_hat, x)
            sparsity_loss = t.norm(encoded_acts, p=1, dim=1).mean() / t.norm(x, p=2, dim=1).mean()
            auxk_loss = self.get_auxiliary_loss(residual, post_relu_acts)
            
            # Total loss
            loss = recon_loss + self.l1_weight * sparsity_loss + self.auxk_alpha * auxk_loss

            if step % 10 == 0 or step == 1:
                l0_sparsity = (encoded_acts != 0).float().sum(dim=1).mean().item()
                print(f"Step {step}/{self.steps} - Phase 2 - "
                    f"Total Loss: {loss.item():.6f}, "
                    f"Recon Loss: {recon_loss.item():.6f}, "
                    f"Sparsity Loss: {sparsity_loss.item():.6f}, "
                    f"l1_weight: {self.l1_weight}, "
                    f"AuxLoss: {auxk_loss.item():.6f}, "
                    f"L0: {l0_sparsity:.2f}, "
                    f"Dead Features: {self.dead_features}")
            
            self.sae_optimizer.zero_grad()
            loss.backward()
            
            self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
                self.ae.decoder.weight,
                self.ae.decoder.weight.grad,
                self.ae.activation_dim,
                self.ae.dict_size,
            )
            t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
            
            self.sae_optimizer.step()
            self.sae_scheduler.step()
            
            # Ensure decoder weights maintain unit norm
            self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
                self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
            )
            
            # Update logging metrics
            self.effective_l0 = (encoded_acts != 0).float().sum(dim=1).mean().item()
            self.recon_loss = recon_loss.item()
            self.sparsity_loss = sparsity_loss.item()
            
        # Phase 3: Joint fine-tuning
        else:
            if step == int(self.steps * self.phase_ratio):
                print("\n" + "="*50)
                print(f"PHASE 3: Joint Fine-tuning with Adaptive Learning Rates and Regularization")
                print("="*50)

                with t.no_grad():
                    # test
                    test_scores = t.linspace(0.0, 10.0, 100, device=self.device)
                    before_k_values = self.probe.map_complexity_to_k(
                        test_scores,
                        min_k=self.min_k,
                        max_k=self.max_k,
                        base_k=self.base_k
                    )
                    before_min_k = before_k_values.min().item()
                    before_max_k = before_k_values.max().item()
                    before_mean_k = before_k_values.float().mean().item()
                    print(f"K value: min={before_min_k}, max={before_max_k}, mean={before_mean_k:.2f}")
                    
                    current_weights = self.probe.weights.data.clone()
                    current_bias = self.probe.bias.data.clone()
                    
                    self.probe.weights.copy_(self.initial_probe_weights)
                    self.probe.bias.copy_(self.initial_probe_bias)
                    
                    weights_match = t.allclose(self.probe.weights, self.initial_probe_weights, rtol=1e-5)
                    bias_match = t.allclose(self.probe.bias, self.initial_probe_bias, rtol=1e-5)
                    
                    after_k_values = self.probe.map_complexity_to_k(
                        test_scores,
                        min_k=self.min_k,
                        max_k=self.max_k,
                        base_k=self.base_k
                    )
                    after_min_k = after_k_values.min().item()
                    after_max_k = after_k_values.max().item()
                    after_mean_k = after_k_values.float().mean().item()
                    print(f"K value: min={after_min_k}, max={after_max_k}, mean={after_mean_k:.2f}")

                # Unfreeze probe parameters
                for param in self.probe.parameters():
                    param.requires_grad = True

            self.current_phase = 3
            
            # probe loss
            pred_complexity = self.probe(x)
            mse_loss = F.mse_loss(pred_complexity, complexity_scores)
            probe_loss = self.probe.compute_loss(x, complexity_scores)

            l2_reg = self.probe.lambda_reg * t.sum(self.probe.weights ** 2) / 2
            
            # deviation loss (prevent probe from deviating too far)
            weights_deviation = t.norm(self.probe.weights - self.initial_probe_weights).to(dtype=t.float32)
            bias_deviation = t.abs(self.probe.bias - self.initial_probe_bias).to(dtype=t.float32)
            deviation_loss = weights_deviation + bias_deviation
            
            k_values = self.probe.map_complexity_to_k(
                pred_complexity,
                min_k=self.min_k,
                max_k=self.max_k,
                base_k=self.base_k
            )
            
            post_relu_acts = F.relu(self.ae.encoder(x - self.ae.b_dec))
            encoded_acts, info = self.ae.encode(x, k_values)
            x_hat = self.ae.decode(encoded_acts)
            
            residual = x - x_hat
            
            self._update_feature_usage(encoded_acts)
            
            # SAE losses
            recon_loss = F.mse_loss(x_hat, x)
            sparsity_loss = t.norm(encoded_acts, p=1, dim=1).mean() / t.norm(x, p=2, dim=1).mean()
            auxk_loss = self.get_auxiliary_loss(residual, post_relu_acts)
            
            sae_loss = recon_loss + self.l1_weight * sparsity_loss + self.auxk_alpha * auxk_loss
            
            # Joint loss
            total_probe_term = probe_loss + self.cur_deviation_weight * deviation_loss
            loss = sae_loss + self.probe_weight * total_probe_term

            if step % 100 == 0:
                l0_sparsity = (encoded_acts != 0).float().sum(dim=1).mean().item()
                probe_rmse = t.sqrt(F.mse_loss(pred_complexity, complexity_scores)).item()
                
                self._print_detailed_losses(
                    step=step, 
                    loss=loss,
                    sae_loss=sae_loss,
                    probe_loss=probe_loss,
                    deviation_loss=deviation_loss,
                    recon_loss=recon_loss,
                    sparsity_loss=sparsity_loss,
                    auxk_loss=auxk_loss,
                    l2_reg=l2_reg,
                    mse_loss=mse_loss,
                    l0_sparsity=l0_sparsity,
                    probe_rmse=probe_rmse,
                    k_values=k_values,
                    pred_complexity=pred_complexity,
                    complexity_scores=complexity_scores
                )
            
            # Backpropagation and optimization
            self.sae_optimizer.zero_grad()
            self.probe_optimizer.zero_grad()
            loss.backward()
            
            # Special gradient handling
            self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
                self.ae.decoder.weight,
                self.ae.decoder.weight.grad,
                self.ae.activation_dim,
                self.ae.dict_size,
            )
            t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
            t.nn.utils.clip_grad_norm_(self.probe.parameters(), 1.0)
            
            self.sae_optimizer.step()
            self.probe_optimizer.step()
            
            self.sae_scheduler.step()
            self.probe_scheduler.step()
            
            # Ensure decoder weights maintain unit norm
            self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
                self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
            )
            
            # Update adaptive deviation weight
            self._update_deviation_weight(probe_loss.item())
            
            # Update logging metrics
            self.effective_l0 = (encoded_acts != 0).float().sum(dim=1).mean().item()
            self.recon_loss = recon_loss.item()
            self.sparsity_loss = sparsity_loss.item()
            self.probe_loss = probe_loss.item()
            self.probe_rmse = t.sqrt(F.mse_loss(pred_complexity, complexity_scores)).item()
        
        return loss.item()

    def _print_detailed_losses(self, step, loss, sae_loss, probe_loss, deviation_loss, 
                            recon_loss, sparsity_loss, auxk_loss, l2_reg, mse_loss,
                            l0_sparsity, probe_rmse, k_values, pred_complexity, complexity_scores):

        k_np = k_values.detach().cpu().numpy()
        k_min = k_np.min()
        k_max = k_np.max()
        k_mean = k_np.mean()
        k_unique = len(np.unique(k_np))
        
        sae_lr = self.sae_optimizer.param_groups[0]['lr']
        probe_lr = self.probe_optimizer.param_groups[0]['lr']
        
        with t.no_grad():
            pred_np = pred_complexity.detach().cpu().numpy()
            true_np = complexity_scores.detach().cpu().numpy()
            errors = pred_np - true_np
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            max_error = np.max(np.abs(errors))
        
        print("\n" + "="*80)
        print(f"STEP {step}/{self.steps} - PHASE 3 - Detailed Loss")
        print("="*80)
        
        print(f"【Total Loss Composition】")
        print(f"  Total Loss: {loss.item():.6f}")
        print(f"  ├─ SAE Loss: {sae_loss.item():.6f} (Weight: 1.0)")
        print(f"  └─ Probe Total: {(probe_loss + self.cur_deviation_weight * deviation_loss).item():.6f} (Weight: {self.probe_weight})")
        print(f"     ├─ Probe Loss: {probe_loss.item():.6f}")
        print(f"     └─ Deviation Loss: {deviation_loss.item():.6f} (Weight: {self.cur_deviation_weight:.4f})")
        
        # SAE
        print(f"\n【SAE Loss Details】")
        print(f"  SAE Loss: {sae_loss.item():.6f}")
        print(f"  ├─ Reconstruction Loss: {recon_loss.item():.6f} (Weight: 1.0)")
        print(f"  ├─ Sparsity Loss: {sparsity_loss.item():.6f} (Weight: {self.l1_weight})")
        print(f"  └─ Auxiliary Loss: {auxk_loss.item():.6f} (Weight: {self.auxk_alpha})")
        
        # probe
        print(f"\n【Probe Loss Details】")
        print(f"  Probe Loss: {probe_loss.item():.6f}")
        print(f"  ├─ MSE Loss: {mse_loss.item():.6f} (Weight: 1.0)")
        print(f"  └─ L2 : {l2_reg.item():.6f} (Weight: {self.probe.lambda_reg})")
        
        # model
        print(f"\n【Model】")
        print(f"  Average Active Features (L0): {l0_sparsity:.2f}")
        print(f"  Dead Features Count: {self.dead_features}")
        print(f"  Probe RMSE: {probe_rmse:.4f}")
        
        # learning rate
        print(f"\n【Learning Rate】")
        print(f"  SAE Learning Rate: {sae_lr:.6f}")
        print(f"  Probe Learning Rate: {probe_lr:.6f}")
        
        # K value
        print(f"\n【K-value Distribution】")
        print(f"  min: {k_min}, max: {k_max}")
        print(f"  mean: {k_mean:.2f} ")
        
        print(f"\n【Prediction】")
        print(f"  Mean Error: {mean_error:.4f}")
        print(f"  Error Standard Deviation: {std_error:.4f}")
        print(f"  Maximum Absolute Error: {max_error:.4f}")
        print("="*80)


    def _update_feature_usage(self, encoded_acts):

        num_tokens_in_step = encoded_acts.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        
        # Mark features that activated in this batch
        active_indices = (encoded_acts != 0).any(dim=0)
        did_fire[active_indices] = True
        
        # Update counters: increment all, reset activated ones to 0
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0
        
        # Count dead features
        self.dead_features = (self.num_tokens_since_fired >= self.dead_feature_threshold).sum().item()
        
    def _update_deviation_weight(self, probe_loss):

        self.probe_loss_history.append(probe_loss)
        if len(self.probe_loss_history) >= 3:
            recent_losses = self.probe_loss_history[-3:]
            if recent_losses[0] > 0:  # Avoid division by zero
                loss_change_rate = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]

                if loss_change_rate > self.loss_stability_threshold:
                    self.cur_deviation_weight = max(0.01, self.cur_deviation_weight * 0.8)
                else:
                    self.cur_deviation_weight = min(0.5, self.cur_deviation_weight * 1.2)
    

    def loss(self, x, step=None, logging=False):

        with t.no_grad():
            complexity_scores = self.probe(x)
            k_values = self.probe.map_complexity_to_k(
                complexity_scores,
                min_k=self.min_k,
                max_k=self.max_k,
                base_k=self.base_k
            )

        post_relu_acts = F.relu(self.ae.encoder(x - self.ae.b_dec))
        encoded_acts, info = self.ae.encode(x, k_values)
        x_hat = self.ae.decode(encoded_acts)
        
        residual = x - x_hat
        
        self.effective_l0 = (encoded_acts != 0).float().sum(dim=1).mean().item()
        
        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = t.norm(encoded_acts, p=1, dim=1).mean() / t.norm(x, p=2, dim=1).mean()
        auxk_loss = self.get_auxiliary_loss(residual, post_relu_acts)
        sae_loss = recon_loss + self.l1_weight * sparsity_loss + self.auxk_alpha * auxk_loss
        
        if not logging:
            return sae_loss
        else:
            losses = {
                "recon_loss": recon_loss.item(),
                "sparsity_loss": sparsity_loss.item(),
                "auxk_loss": auxk_loss.item(),
                "sae_loss": sae_loss.item(),
                "probe_loss": self.probe_loss if hasattr(self, 'probe_loss') else -1,
                "probe_rmse": self.probe_rmse if hasattr(self, 'probe_rmse') else -1,
                "loss": sae_loss.item()
            }
            
            from collections import namedtuple
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x, x_hat, encoded_acts, losses
            )
            
    def get_logging_parameters(self):
        return {
            "effective_l0": self.effective_l0,
            "dead_features": self.dead_features,
            "probe_loss": self.probe_loss,
            "recon_loss": self.recon_loss,
            "sparsity_loss": self.sparsity_loss,
            "probe_rmse": self.probe_rmse,
            "current_phase": self.current_phase,
            "deviation_weight": self.cur_deviation_weight,
            "pre_norm_auxk_loss": self.pre_norm_auxk_loss
        }

    def save_model(self, save_dir):

        os.makedirs(save_dir, exist_ok=True)

        ae_state_dict = self.ae.state_dict()
        t.save(ae_state_dict, os.path.join(save_dir, "ae.pt"))
        
        probe_state_dict = self.probe.state_dict()
        t.save(probe_state_dict, os.path.join(save_dir, "probe.pt"))

        config = {
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "base_k": self.base_k,
            "min_k": self.min_k,
            "max_k": self.max_k,
            "l1_weight": self.l1_weight,
            "probe_weight": self.probe_weight,
            "trainer_type": "DynamicTopKTrainer"
        }
        
        with open(os.path.join(save_dir, "dynamic_config.json"), "w") as f:
            json.dump(config, f, indent=4)
        
        print(f"all saved to {save_dir}")
    
    @property
    def config(self):
        return {
            "trainer_class": "DynamicTopKTrainer",
            "dict_class": "AutoEncoderDynamicTopK",
            "sae_lr": self.sae_lr,
            "probe_lr": self.probe_lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "base_k": self.base_k,
            "min_k": self.min_k,
            "max_k": self.max_k,
            "l1_weight": self.l1_weight,
            "probe_weight": self.probe_weight,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "current_phase": self.current_phase,
        }