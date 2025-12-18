import torch
from collections import OrderedDict

from xreflection.utils.registry import MODEL_REGISTRY
from xreflection.models.base_model import BaseModel
from xreflection.archs.vlm.utils import *
from transformers import AutoModel, AutoTokenizer

@MODEL_REGISTRY.register()
class GuidedUNetModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)

        # Losses (initialized in setup)
        self.cri_pix = None
        
        self.vlm_path = 'OpenGVLab/InternVL2_5-2B'
        self.vlm_tokenizer = AutoTokenizer.from_pretrained(self.vlm_path, trust_remote_code=True, use_fast=False)
        self.vlm_model = AutoModel.from_pretrained(
            self.vlm_path,
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
        )
        for param in self.vlm_model.parameters():
            param.requires_grad = False
        self.vlm_model.eval()
        
        self.vlm_model.img_context_token_id = self.vlm_tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.vlm_prompt = DEFAULT_VLM_PROMPT
        self.vlm_tokens_per_patch = 256
        
        self.vlm_transform = build_transform()
        self._input_id_cache = {}
        

    def setup_losses(self):
        """Setup loss functions"""
        from xreflection.losses import build_loss
        if not hasattr(self, 'cri_pix') or self.cri_pix is None:
            if self.opt['train'].get('pixel_opt'):
                self.cri_pix = build_loss(self.opt['train']['pixel_opt'])
                
    def _get_input_ids(self, num_patches):
        # Input IDs 캐싱 (매번 토크나이징 방지)
        if num_patches in self._input_id_cache:
            input_ids = self._input_id_cache[num_patches]
            if input_ids.device == self.device:
                return input_ids
        
        img_token_len = self.vlm_tokens_per_patch * num_patches
        question_text = self.vlm_prompt.replace('<image>', '<IMG_CONTEXT>' * img_token_len)
        input_ids = self.vlm_tokenizer(question_text, return_tensors='pt').input_ids.to(self.device)
        self._input_id_cache[num_patches] = input_ids
        return input_ids
                
    @torch.no_grad()
    def _extract_guide_emb(self, image_tensor):
        """
        이미지 텐서 1장 -> VLM Embedding 추출
        """
        image = image_tensor.detach()
        pil_img = TF.ToPILImage()(image.cpu().clamp(0, 1))
        
        # Dynamic Preprocess (InternVL 로직)
        images = dynamic_preprocess(pil_img)
        
        # VLM Transform 적용
        pixel_values = [self.vlm_transform(img) for img in images]
        pixel_values = torch.stack(pixel_values).to(self.device, dtype=torch.bfloat16)

        num_patches = pixel_values.size(0)
        image_flags = torch.ones(num_patches, device=self.device)
        input_ids = self._get_input_ids(num_patches)

        # Inference
        outputs = self.vlm_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            image_flags=image_flags,
            output_hidden_states=True,
            return_dict=True,
        )
        # 마지막 레이어, 마지막 토큰
        last_hidden_state = outputs.hidden_states[-1]
        guide_emb = last_hidden_state[:, -1, :] # (1, 2048)
            
        return guide_emb.float() # U-Net은 float32 사용
    
    def _extract_guide_emb_batch(self, batch_images):
        """
        배치 전체 처리 (Loop 사용)
        InternVL은 입력 사이즈가 가변적이라 Batch 처리가 까다로우므로 Loop가 안전함
        """
        emb_list = []
        for i in range(batch_images.size(0)):
            emb = self._extract_guide_emb(batch_images[i])
            emb_list.append(emb)
        return torch.cat(emb_list, dim=0)

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
        guide_emb = self._extract_guide_emb_batch(inp)
        target_t = batch['target_t']
    
        # Forward pass
        output_r = self.net_g(inp, guide_emb)
        output_t = inp - output_r

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
            guide_emb = self._extract_guide_emb_batch(inp)
            r = model(inp, guide_emb)
            self.output = [inp - r, r]
