import torch
import torch.nn as nn
from DAMP_model import DAMP_models

class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.seq_len = args.seq_len
        self.embeddings_dim = args.embeddings_dim
        self.num_features = args.num_features
        self.pretrain = args.pretrain
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        self.net = DAMP_models[args.model](
            seq_len=self.seq_len,
            num_features=self.num_features,
            pretrain = args.pretrain,

        )
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)
        self.cfg_phychem = args.cfg_phychem
        self.cfg_family = args.cfg_family

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels,family_embedding):
        
        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels,family_embedding)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2)).mean()
        return loss

    @torch.no_grad()
    def generate(self, condition,family_embedding):
        device = condition.device
        batch = condition.size(0)
        z = self.noise_scale * torch.randn(
            batch, self.seq_len, self.embeddings_dim, device=device
        )
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, 1, 1, 1).expand(-1, batch, 1, 1) 
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"Unknown method: {self.method}")

        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, condition, family_embedding)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], condition, family_embedding)
        
        return z


    @torch.no_grad()
    def _forward_sample(self, z, t, condition,family_embedding):

        null_phychem = self.net.y_embedder.null_token.unsqueeze(0).expand(z.shape[0], -1)
        null_family = self.net.f_embedder.null_token.unsqueeze(0).expand(z.shape[0], -1,-1)

        # x_cond_phychem = self.net(z, t.flatten(), condition, null_family,)
        # v_cond_phychem = (x_cond_phychem - z) / (1.0 - t).clamp_min(self.t_eps)

        # x_cond_family = self.net(z, t.flatten(), null_phychem, family_embedding,)
        # v_cond_family = (x_cond_family - z) / (1.0 - t).clamp_min(self.t_eps)
        
        # x_cond_both = self.net(z, t.flatten(), condition, family_embedding)
        # v_cond_both = (x_cond_both - z) / (1.0 - t).clamp_min(self.t_eps)
        
        x_un = self.net(z, t.flatten(), null_phychem,null_family)
        v_un = (x_un - z) / (1.0 - t).clamp_min(self.t_eps)
        
        return v_un
        low, high = self.cfg_interval
        
        interval_mask = (t < high) & ((low == 0) | (t > low))
        
        cfg_phychem_interval = torch.where(
            interval_mask, 
            self.cfg_phychem, 
            0
        )
        cfg_family_interval = torch.where(
            interval_mask, 
            self.cfg_family, 
            0
        )
        interaction = v_cond_both - v_cond_phychem - v_cond_family + v_un
        return (v_un 
            + cfg_phychem_interval * (v_cond_phychem - v_un)
            + cfg_family_interval * (v_cond_family - v_un)
            + cfg_phychem_interval * cfg_family_interval * interaction)


    @torch.no_grad()
    def _euler_step(self, z, t, t_next, condition,family_embedding):
        v_pred = self._forward_sample(z, t, condition,family_embedding=family_embedding)
        z_next = z + (t_next - t) * v_pred
        return z_next


    @torch.no_grad()
    def _heun_step(self, z, t, t_next, condition,family_embedding):
        v_pred_t = self._forward_sample(z, t, condition,family_embedding=family_embedding)
        z_next_euler = z + (t_next - t) * v_pred_t
        
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, condition,family_embedding= family_embedding)
        
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        
        return z_next