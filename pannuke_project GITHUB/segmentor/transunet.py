import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.models.vision_transformer import vit_small_patch8_224 as ViT
from timm.models.vision_transformer import PatchEmbed

class TransUNet(nn.Module):
    def __init__(self, in_chans: int = 3, num_classes: int = 6, num_organs: int = 19, pretrained_vit: bool = True):
        super().__init__()

        self.encoder = ViT(pretrained=pretrained_vit, in_chans=in_chans)
        self.D = self.encoder.embed_dim
        ps = self.encoder.patch_embed.patch_size
        self.ps = ps[0] if isinstance(ps, tuple) else ps

        self.encoder.patch_embed = PatchEmbed(
            img_size=None,
            patch_size=ps,
            in_chans=in_chans,
            embed_dim=self.D,
            flatten=False         
        )

        self.encoder.head = nn.Identity()

        self.cls_token = self.encoder.cls_token
        self.pos_embed = self.encoder.pos_embed
        self.pos_drop  = self.encoder.pos_drop
        self.blocks    = self.encoder.blocks
        self.norm      = self.encoder.norm

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.D,   self.D//2, 2, 2), nn.BatchNorm2d(self.D//2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.D//2, self.D//4, 2, 2), nn.BatchNorm2d(self.D//4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.D//4, self.D//8, 2, 2), nn.BatchNorm2d(self.D//8), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.D//8, 64, 2, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # <--- 最後一層 out_channels=64
        )
        self.seg_head = nn.Conv2d(64, 2, kernel_size=1)
        self.type_head = nn.Conv2d(64, num_classes, kernel_size=1)

        self.organ_head = nn.Linear(self.D, num_organs)

    def _to_4d(self, patches3or4: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if patches3or4.ndim == 4:
            return patches3or4.contiguous()
        B, N, D = patches3or4.shape
        Hp = H // self.ps
        Wp = W // self.ps
        return patches3or4.transpose(1, 2).contiguous().view(B, D, Hp, Wp)

    def _interp_pos(self, B: int, Hp: int, Wp: int) -> torch.Tensor:
        cls = self.cls_token.expand(B, -1, -1)          
        pos = self.pos_embed[:, 1:]                     
        orig = int(pos.size(1) ** 0.5)
        p = pos.reshape(1, orig, orig, -1).permute(0,3,1,2)    
        p = F.interpolate(p, size=(Hp, Wp), mode='bicubic', align_corners=False)
        p = p.permute(0,2,3,1).reshape(1, Hp*Wp, -1)         
        return torch.cat((cls, p.expand(B, -1, -1)), dim=1)    

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        patches4d = self._to_4d(self.encoder.patch_embed(x), H, W)  
        _, _, Hp, Wp = patches4d.shape

        patch_tokens = patches4d.flatten(2).transpose(1, 2)        
        cls_tokens   = self.cls_token.expand(B, -1, -1)            
        tokens       = torch.cat([cls_tokens, patch_tokens], dim=1)

        pos_emb = self._interp_pos(B, Hp, Wp)
        if pos_emb.size(1) == tokens.size(1) + 1:
            pos_emb = pos_emb[:, 1:, :]
        elif pos_emb.size(1) != tokens.size(1):
            if pos_emb.size(1) + 1 == tokens.size(1):
                zeros = torch.zeros((B, 1, self.D), device=pos_emb.device, dtype=pos_emb.dtype)
                pos_emb = torch.cat([zeros, pos_emb], dim=1)
            else:
                pos_emb = pos_emb[:, :tokens.size(1), :]
        tokens = tokens + pos_emb
        tokens = self.pos_drop(tokens)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        cls_feat = tokens[:, 0, :]                    

        cls_feat = torch.nan_to_num(cls_feat)
        cls_feat = torch.clamp(cls_feat, -1e4, 1e4)
        organ_logits = self.organ_head(cls_feat)       

        patch_len = tokens[:, 1:, :].shape[1]
        HpWp = int(patch_len ** 0.5)
        if HpWp * HpWp != patch_len:
            raise RuntimeError(f"patch tokens: {patch_len}")
        feat = tokens[:, 1:, :].transpose(1, 2).reshape(B, self.D, HpWp, HpWp)
        dec_feat = self.decoder(feat)

        seg_logits  = self.seg_head(dec_feat)       
        type_logits = self.type_head(dec_feat)       
        return seg_logits, type_logits, organ_logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        patches4d = self._to_4d(self.encoder.patch_embed(x), H, W)
        _, _, Hp, Wp = patches4d.shape

        patch_tokens = patches4d.flatten(2).transpose(1, 2)
        cls_tokens   = self.cls_token.expand(B, -1, -1)
        tokens       = torch.cat([cls_tokens, patch_tokens], dim=1)

        pos_emb = self._interp_pos(B, Hp, Wp)
        if pos_emb.size(1) == tokens.size(1) + 1:
            pos_emb = pos_emb[:, 1:, :]
        elif pos_emb.size(1) != tokens.size(1):
            if pos_emb.size(1) + 1 == tokens.size(1):
                zeros = torch.zeros((B, 1, self.D), device=pos_emb.device, dtype=pos_emb.dtype)
                pos_emb = torch.cat([zeros, pos_emb], dim=1)
            else:
                pos_emb = pos_emb[:, :tokens.size(1), :]
        tokens = tokens + pos_emb
        tokens = self.pos_drop(tokens)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        cls_feat = tokens[:, 0, :]                   

        cls_feat = torch.nan_to_num(cls_feat)
        cls_feat = torch.clamp(cls_feat, -1e4, 1e4)
        organ_logits = self.organ_head(cls_feat)      

        patch_len = tokens[:, 1:, :].shape[1]
        HpWp = int(patch_len ** 0.5)
        if HpWp * HpWp != patch_len:
            raise RuntimeError(f"patch tokens: {patch_len}")
        feat = tokens[:, 1:, :].transpose(1, 2).reshape(B, self.D, HpWp, HpWp)
        dec_feat = self.decoder(feat)
        seg_logits = self.seg_head(dec_feat)
        type_logits = self.type_head(dec_feat)
        return seg_logits, type_logits, organ_logits  
