import timm
import torch.nn as nn


class ViTBackbone(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, return_cls=False):
        """
        Args:
            model_name (str): 'vit_base_patch16_224' or 'vit_large_patch16_224'
            pretrained (bool): load pretrained weights from timm
            return_cls (bool): if True, return CLS token instead of patch tokens
        """
        super().__init__()
        self.return_cls = return_cls

        # Load model with no classification head
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # Get dimensions
        self.embed_dim = self.vit.embed_dim  # 768 for B/16, 1024 for L/16
        self.patch_size = self.vit.patch_embed.patch_size

    def forward(self, x):
        # Features includes CLS + patch tokens: [B, 1 + N, D]
        x = self.vit.forward_features(x)

        if self.return_cls:
            return x[:, 0]  # [B, D] – CLS token
        else:
            patch_tokens = x[:, 1:]  # [B, N, D]
            B, N, D = patch_tokens.shape
            H = W = int(N ** 0.5)
            return patch_tokens.view(B, D, H, W)  # [B, H, W, D]