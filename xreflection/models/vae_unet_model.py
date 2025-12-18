import torch
from collections import OrderedDict

from xreflection.utils.registry import MODEL_REGISTRY
from xreflection.models.base_model import BaseModel
from diffusers import AutoencoderKL

@MODEL_REGISTRY.register()
class LatentUNetModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # VAE - freeze weights
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Losses (initialized in setup)
        self.cri_pix = None        

    def setup_losses(self):
        """Setup loss functions"""
        from xreflection.losses import build_loss
        if not hasattr(self, 'cri_pix') or self.cri_pix is None:
            if self.opt['train'].get('pixel_opt'):
                self.cri_pix = build_loss(self.opt['train']['pixel_opt'])
                
    def _encode(self, x):
        posterior = self.vae.encode(x)
        z = posterior.latent_dist.sample()
        return z

    def _decode(self, z):
        x = self.vae.decode(z).sample
        return x

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

        # Forward pass
        z_inp = self._encode(inp * 2.0 - 1.0)   # scale to [-1, 1]: VAE input range
        z_t = self.net_g(z_inp)
        output_t = self._decode(z_t)
        output_t = (output_t + 1.0) / 2.0  # scale to [0, 1]

        # Calculate losses
        loss_dict = OrderedDict()

        # Pixel loss
        l_g_pix_t = self.cri_pix(output_t, target_t)

        # Total loss
        loss_dict['l_g_pix_t'] = l_g_pix_t
        
        l_g_total = l_g_pix_t

        # Log losses
        for name, value in loss_dict.items():
            self.log(f'train/{name}', value, prog_bar=True, sync_dist=True)

        # Store outputs for visualization
        self.last_inp = inp
        self.last_output_clean = output_t
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
            z_inp = self._encode(inp * 2.0 - 1.0)
            z_t = model(z_inp)
            t = (self._decode(z_t) + 1.0) / 2.0
            
            self.output = [t, inp - t]
