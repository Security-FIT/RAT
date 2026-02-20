import torch
import torch.distributed as dist
from tqdm import tqdm

from trainers.BaseFFTrainer import BaseFFTrainer


class FFPairTrainer(BaseFFTrainer):
    def __init__(
        self,
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(model, device)

    def train_epoch(self, train_dataloader) -> tuple[list[float], list[float]]:
        """
        Train the model on the given dataloader for one epoch
        Uses the optimizer and loss function defined in the constructor

        param train_dataloader: Dataloader loading the training data
        return: Tuple(lists of accuracies, list of losses)
        """
        # For accuracy computation in the epoch
        losses = []
        accuracies = []

        # Training loop
        for _, gt, test, label in tqdm(train_dataloader):
            gt = gt.to(self.device)
            test = test.to(self.device)
            label = label.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device.split(':')[0] if ":" in self.device else self.device, dtype=torch.bfloat16):
                logits, probs = self.model(gt, test)
                loss = self.lossfn(logits, label.long())

            # Loss and backpropagation
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            predicted = torch.argmax(probs, 1)
            correct = (predicted == label).sum().item()
            accuracy = correct / len(label)

            losses.append(loss.item())
            accuracies.append(accuracy)

        return accuracies, losses

    def val_epoch(
        self, val_dataloader, save_scores=False
    ) -> tuple[list[float], list[float], list[float], list[int], list[str]]:
        losses = []
        labels = []
        scores = []
        predictions = []
        file_names = []

        for file_name, gt, test, label in tqdm(val_dataloader):
            # print(f"Validation batch {i+1} of {len(val_dataloader)}")

            # Sanity check: Verify corruption is applied to the reference tensor
            # if hasattr(val_dataloader.dataset, "ref_snr") and val_dataloader.dataset.ref_snr is not None:
            #     # Check if the reference tensor is not silent (unless ref_silence is True)
            #     if not (hasattr(val_dataloader.dataset, "ref_silence") and val_dataloader.dataset.ref_silence):
            #          # Simple check: print mean and std of the first batch
            #          if len(losses) == 0:
            #              print(f"Reference tensor stats - Mean: {gt.mean().item()}, Std: {gt.std().item()}")

            gt = gt.to(self.device)
            test = test.to(self.device)
            label = label.to(self.device)

            with torch.autocast(device_type=self.device.split(':')[0] if ":" in self.device else self.device, dtype=torch.bfloat16):
                logits, probs = self.model(gt, test)
                loss = self.lossfn(logits, label.long())

            predictions.extend(torch.argmax(probs, 1).tolist())

            if save_scores:
                file_names.extend(file_name)
            losses.append(loss.item())
            labels.extend(label.tolist())
            scores.extend(probs[:, 0].tolist())

        # for name, label, score, prediction in zip(file_names, labels, scores, predictions):
        #     print(f"File: {name}, Score: {score}, Label: {label}, Prediction: {prediction}")

        return losses, labels, scores, predictions, file_names

    def grad_ratio_emb(self, val_dataloader, eps=1e-12):
        self.model.eval()
        dev = self.device
        dev_type = "cuda" if "cuda" in str(dev) else "cpu"

        raw_model = getattr(self.model, "module", self.model)

        # make extractor cheap
        raw_model.extractor.finetune = False

        # IMPORTANT: no parameter grads (saves memory)
        for p in raw_model.parameters():
            p.requires_grad_(False)

        sum_ref = torch.zeros((), device=dev, dtype=torch.float32)
        sum_test = torch.zeros((), device=dev, dtype=torch.float32)
        sumsq_ref = torch.zeros((), device=dev, dtype=torch.float32)
        sumsq_test = torch.zeros((), device=dev, dtype=torch.float32)
        count_samples = torch.zeros((), device=dev, dtype=torch.float32)

        show_progress = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
        iterator = tqdm(val_dataloader) if show_progress else val_dataloader

        for _, gt, test, label in iterator:
            gt = gt.to(dev, non_blocking=True)
            test = test.to(dev, non_blocking=True)
            label = label.to(dev, non_blocking=True).long()
            B = gt.size(0)
            count_samples += B

            with torch.no_grad():
                with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type=="cuda")):
                    emb_gt = raw_model.extractor.extract_features(gt)
                    emb_test = raw_model.extractor.extract_features(test)

            emb_gt = emb_gt.detach().requires_grad_(True)
            emb_test = emb_test.detach().requires_grad_(True)

            with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type=="cuda")):
                logits, _ = raw_model.forward_from_embedding(emb_gt, emb_test)  # NO intermediates
                loss = self.lossfn(logits, label)

            g_ref, g_test = torch.autograd.grad(
                loss, [emb_gt, emb_test], retain_graph=False, create_graph=False, allow_unused=False
            )

            # per-sample RMS over L,T,F for [L,B,T,F]
            r_ref = g_ref.detach().float().pow(2).mean(dim=(0,2,3)).sqrt()   # [B]
            r_test = g_test.detach().float().pow(2).mean(dim=(0,2,3)).sqrt() # [B]

            sum_ref += r_ref.sum()
            sum_test += r_test.sum()
            sumsq_ref += (r_ref * r_ref).sum()
            sumsq_test += (r_test * r_test).sum()

        
        # all-reduce across ranks
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_ref, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_test, op=dist.ReduceOp.SUM)
            dist.all_reduce(sumsq_ref, op=dist.ReduceOp.SUM)
            dist.all_reduce(sumsq_test, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_samples, op=dist.ReduceOp.SUM)

        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0: # Only print from rank 0
            mean_ref = (sum_ref / count_samples).item()
            mean_test = (sum_test / count_samples).item()

            var_ref = ((sumsq_ref / count_samples) - mean_ref**2).clamp_min(0.0).item()
            var_test = ((sumsq_test / count_samples) - mean_test**2).clamp_min(0.0).item()
            std_ref = var_ref ** 0.5
            std_test = var_test ** 0.5

            ratio = mean_ref / mean_test

            print(f"Reference gradient norms - Mean: {mean_ref}, Std: {std_ref}")
            print(f"Test gradient norms - Mean: {mean_test}, Std: {std_test}")
            print(f"Average gradient norm ratio (reference/test): {ratio}")

    @staticmethod
    def generate_noise_ref(gt_waveform):
        noise = torch.randn_like(gt_waveform)
        # Calculate energy per sample in batch
        signal_power = torch.mean(gt_waveform ** 2, dim=1, keepdim=True)
        noise_power = torch.mean(noise ** 2, dim=1, keepdim=True)
        scale = torch.sqrt(signal_power / (noise_power + 1e-8))
        return noise * scale

    def delta_trace(self, val_dataloader, keys, eps=1e-12):
        self.model.eval()
        dev = self.device
        dev_type = "cuda" if "cuda" in str(dev) else "cpu"

        raw_model = getattr(self.model, "module", self.model)
        raw_model.extractor.finetune = False

        # Prepare keys for tracking both metrics
        # Standard keys track "vs Energy-Matched Noise"
        # _zero keys track "vs Zero/Silent Ref"
        sum_D = {k: torch.zeros((), device=dev, dtype=torch.float32) for k in keys}
        sum_D_zero = {k: torch.zeros((), device=dev, dtype=torch.float32) for k in keys}
        
        count_samples = torch.zeros((), device=dev, dtype=torch.float32)

        sum_abs_dm = torch.zeros((), device=dev, dtype=torch.float32)
        sum_abs_dm_zero = torch.zeros((), device=dev, dtype=torch.float32)
        sum_abs_m  = torch.zeros((), device=dev, dtype=torch.float32)

        sum_ratio = torch.zeros((), device=dev, dtype=torch.float32)
        sumsq_ratio = torch.zeros((), device=dev, dtype=torch.float32)
        ratio_count = torch.zeros((), device=dev, dtype=torch.float32)

        show_progress = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
        iterator = tqdm(val_dataloader) if show_progress else val_dataloader

        for _, gt, test, _label in iterator:
            gt = gt.to(dev, non_blocking=True)
            test = test.to(dev, non_blocking=True)
            B = gt.shape[0]
            count_samples += float(B)

            gt2 = self.generate_noise_ref(gt)      # Noise reference (energy matched)
            gt3 = torch.zeros_like(gt)             # Silent reference

            with torch.no_grad():
                with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=(dev_type=="cuda")):
                    emb_test = raw_model.extractor.extract_features(test)
                    emb_gtA  = raw_model.extractor.extract_features(gt)
                    emb_gtB  = raw_model.extractor.extract_features(gt2)
                    emb_gtC  = raw_model.extractor.extract_features(gt3)

                    _, _, interA = raw_model.forward_from_embedding(emb_gtA, emb_test, return_intermediate=True)
                    _, _, interB = raw_model.forward_from_embedding(emb_gtB, emb_test, return_intermediate=True)
                    _, _, interC = raw_model.forward_from_embedding(emb_gtC, emb_test, return_intermediate=True)

            attn = interA["attn_map"].float().contiguous()  # [T, L*B, F]
            mlp  = interA["mlp_out"].float().contiguous()   # [T, L*B, F]

            T, LB, F = attn.shape
            assert LB % B == 0, f"LB={LB} not divisible by B={B}"
            L = LB // B

            attn4 = attn.view(T, L, B, F)  # [T, L, B, F]
            mlp4  = mlp.view(T, L, B, F)

            # per-sample RMS over T,L,F -> [B]
            attn_rms = attn4.pow(2).mean(dim=(0,1,3)).sqrt()
            mlp_rms  = mlp4.pow(2).mean(dim=(0,1,3)).sqrt()

            ratio = attn_rms / (mlp_rms + eps)  # [B]

            sum_ratio += ratio.sum()
            sumsq_ratio += (ratio * ratio).sum()
            ratio_count += float(B)

            logA = interA["logits"].float()  # [B,2]
            logB = interB["logits"].float()  # [B,2]
            logC = interC["logits"].float()  # [B,2]

            mA = (logA[:,0] - logA[:,1])
            mB = (logB[:,0] - logB[:,1])
            mC = (logC[:,0] - logC[:,1])

            sum_abs_dm += (mA - mB).abs().sum()        # vs Noise
            sum_abs_dm_zero += (mA - mC).abs().sum()   # vs Zero
            sum_abs_m  += mA.abs().sum()

            for k in keys:
                a = interA[k].float()
                
                # Flatten A once
                if a.ndim == 4 and a.shape[1] == B:
                    aB = a.permute(1,0,2,3).contiguous().view(B, -1)
                elif a.ndim == 3:                             
                    T, LB, F = a.shape
                    L = LB // B
                    aB = a.view(T, L, B, F).permute(2,1,0,3).contiguous().view(B, -1)
                elif a.ndim == 2 and a.shape[0] == B:
                    aB = a.contiguous().view(B, -1)
                else:
                    aB = a.view(B, -1)

                denom_norm = aB.norm(p=2, dim=1).clamp_min(eps) # [B]

                # Compare against B (Noise) and C (Zero)
                targets = [(interB, sum_D), (interC, sum_D_zero)]

                for inter_target, sum_target in targets:
                    b = inter_target[k].float()
                    
                    if b.ndim == 4 and b.shape[1] == B:
                        bB = b.permute(1,0,2,3).contiguous().view(B, -1)
                    elif b.ndim == 3:
                        T, LB, F = b.shape
                        L = LB // B
                        bB = b.view(T, L, B, F).permute(2,1,0,3).contiguous().view(B, -1)
                    elif b.ndim == 2 and b.shape[0] == B:
                        bB = b.contiguous().view(B, -1)
                    else:
                        bB = b.view(B, -1)

                    numerator = (aB - bB).norm(p=2, dim=1)
                    D = (numerator / denom_norm).mean()
                    sum_target[k] += D * B

        torch.cuda.synchronize()  # Ensure all computations are finished before printing results
        # all-reduce across ranks
        if dist.is_available() and dist.is_initialized():
            # DIAGNOSTIC: Check which ranks are actually in this group
            my_rank = torch.tensor([dist.get_rank()], device=dev)
            rank_list = [torch.zeros_like(my_rank) for _ in range(dist.get_world_size())]
            dist.all_gather(rank_list, my_rank)
            visible_ranks = sorted([r.item() for r in rank_list])
            if dist.get_rank() == 0:
                print(f"DDP Group Members (expected 8?): {visible_ranks} (Total: {len(visible_ranks)})", flush=True)

            if dist.get_rank() == 0:
                print(f"[rank {dist.get_rank()}] local count_samples={count_samples.item()}", flush=True)
            
            # Pack values for single reduction
            # [count, sum_abs_dm, sum_abs_dm_zero, sum_abs_m, count_ratio, sum_ratio, sumsq_ratio]
            # + [sum_D[k]...] + [sum_D_zero[k]...]
            header = [count_samples, sum_abs_dm, sum_abs_dm_zero, sum_abs_m, ratio_count, sum_ratio, sumsq_ratio]
            packed = header + [sum_D[k] for k in keys] + [sum_D_zero[k] for k in keys]
            packed_tensor = torch.stack(packed)
            dist.all_reduce(packed_tensor, op=dist.ReduceOp.SUM)
            
            # Unpack
            count_samples = packed_tensor[0]
            sum_abs_dm = packed_tensor[1]
            sum_abs_dm_zero = packed_tensor[2]
            sum_abs_m = packed_tensor[3]
            ratio_count = packed_tensor[4]
            sum_ratio = packed_tensor[5]
            sumsq_ratio = packed_tensor[6]
            
            offset = 7
            for i, k in enumerate(keys):
                sum_D[k] = packed_tensor[offset + i]
            
            offset += len(keys)
            for i, k in enumerate(keys):
                sum_D_zero[k] = packed_tensor[offset + i]

        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0: # Only print from rank 0
            print(f"Activation delta trace mode, mean normalized L2 change on {count_samples} samples:")
            print(f"{'Key':<15} | {'vs NOISE':<15} | {'vs ZERO':<15}")
            print("-" * 50)

            for key in keys:
                meanD = (sum_D[key] / count_samples.clamp_min(1.0)).item()
                meanD_z = (sum_D_zero[key] / count_samples.clamp_min(1.0)).item()
                print(f"{key:15s} | {meanD:.3e}       | {meanD_z:.3e}")

            print("-" * 50)
            delta_m = (sum_abs_dm / (count_samples + eps)).item()
            delta_m_norm = (sum_abs_dm / (sum_abs_m + eps)).item()
            print(f"Δmargin (NOISE): {delta_m:.3e}  (normalized: {delta_m_norm:.3e})")
            
            delta_m_z = (sum_abs_dm_zero / (count_samples + eps)).item()
            delta_m_z_norm = (sum_abs_dm_zero / (sum_abs_m + eps)).item()
            print(f"Δmargin (ZERO) : {delta_m_z:.3e}  (normalized: {delta_m_z_norm:.3e})")

            mean_ratio = (sum_ratio / (ratio_count + eps)).item()
            var_ratio = (sumsq_ratio / (ratio_count + eps) - mean_ratio**2).clamp_min(0.0).item()
            std_ratio = var_ratio ** 0.5
            print(f"R_attn/mlp (RMS): mean={mean_ratio:.3e}, std={std_ratio:.3e}")


    def ref_sensitivity(self, val_dataloader):
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            print("Computing reference sensitivity (grad ratio and delta trace)...")
        self.grad_ratio_emb(val_dataloader)
        self.delta_trace(val_dataloader, keys=["attn_map", "mlp_out", "combined", "residual", "final_out", "emb", "logits"])
