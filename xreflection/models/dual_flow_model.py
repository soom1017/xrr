import torch
import torch.nn as nn
from collections import OrderedDict

from xreflection.utils.registry import MODEL_REGISTRY
from xreflection.models.base_model import BaseModel


@MODEL_REGISTRY.register()
class DualHeadFMModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # Losses (initialized in setup)
        self.cri_score = None
        self.cri_recons = None
        self.cri_pix = None

    def setup_losses(self):
        """Setup loss functions"""
        from xreflection.losses import build_loss
        if not hasattr(self, 'cri_pix') or self.cri_pix is None:
            if self.opt['train'].get('pixel_opt'):
                self.cri_pix = build_loss(self.opt['train']['pixel_opt'])
                
        self.cri_score = nn.MSELoss()
        self.cri_recons = nn.L1Loss()

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch (dict): Input batch containing 'input', 'target_t', 'target_r'.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Total loss.
        """
        # Get inputs
        inp = batch['input']
        target_t = batch['target_t']
        target_r = batch['target_r']

        # sample t ~ U(0,1)
        B = inp.size(0)
        t = torch.rand(B, device=inp.device)
        t_view = t.view(B, 1, 1, 1)
        
        # ---- start state (x0): blended image for T, zero for R ----
        xT0 = inp
        xR0 = torch.zeros_like(inp)
        x0 = torch.cat([xT0, xR0], dim=1)   # (B, 6, H, W)
        # ---- target state (x1) ----
        x1 = torch.cat([target_t, target_r], dim=1)
        # ---- rectified flow straight interpolation ----
        x_t = (1.0 - t_view) * x0 + t_view * x1
        
        # Forward pass: predict vector fields
        v_t, v_r = self.net_g(x_t, t, inp)
        
        # ---- one-step Euler surrogate for x_hat at t=1 ----
        v = torch.cat([v_t, v_r], dim=1)
        x_hat = x_t + (1.0 - t_view) * v
        xT_hat, xR_hat = x_hat[:, :3], x_hat[:, 3:]

        # Calculate losses
        loss_dict = OrderedDict()

        l_g_pix_t = self.cri_pix(xT_hat, target_t)
        l_g_pix_r = self.cri_pix(xR_hat, target_r)
        
        l_g_score_t = self.cri_score(v_t, target_t - inp)
        l_g_score_r = self.cri_score(v_r, target_r)
        l_g_recons = self.cri_recons(xT_hat + xR_hat, inp)

        # Total loss
        loss_dict['l_g_pix_t'] = l_g_pix_t
        loss_dict['l_g_pix_r'] = l_g_pix_r
        loss_dict['l_g_score_t'] = l_g_score_t
        loss_dict['l_g_score_r'] = l_g_score_r
        loss_dict['l_g_recons'] = l_g_recons
        
        l_g_total = (l_g_pix_t + l_g_pix_r) * 0.1 + l_g_score_t + l_g_score_r + l_g_recons * 0.5
        # Log losses
        for name, value in loss_dict.items():
            self.log(f'train/{name}', value, prog_bar=True, sync_dist=True)

        # Store outputs for visualization
        self.last_inp = inp
        self.last_output_clean = xT_hat
        self.last_output_reflection = xR_hat
        self.last_target_t = target_t

        return l_g_total

    def configure_optimizer_params(self):
        """Configure optimizer parameters.
        
        Returns:
            dict: Optimizer configuration.
        """
        train_opt = self.opt['train']

        # Get all network parameters
        params = list(self.net_g.parameters())

        # Get optimizer configuration
        optim_type = train_opt['optim_g']['type']
        optim_config = {k: v for k, v in train_opt['optim_g'].items() if k != 'type'}

        return {
            'optim_type': optim_type,
            'params': params,
            **optim_config,
        }

    def testing(self, inp):
        if self.use_ema:
            model = self.ema_model
        else:
            model = self.net_g
        with torch.no_grad():
            xT0 = inp
            xR0 = torch.zeros_like(inp)
            x = torch.cat([xT0, xR0], dim=1)  # (B,6,H,W)
            
            # integrate from t=0 -> 1
            n_steps = self.opt['val'].get('n_steps', 50)
            dt = 1.0 / n_steps
            B = inp.size(0)

            for k in range(n_steps):
                tk = torch.full((B,), k * dt, device=inp.device)
                vT, vR = model(x, tk, inp)
                v = torch.cat([vT, vR], dim=1)
                x = x + dt * v

            xT, xR = x[:, :3], x[:, 3:]
            self.output = [xT, xR]
            