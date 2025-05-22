import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import cfg



class HACO(nn.Module):
    def __init__(self):
        super(HACO, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)

        # Load modules
        self.backbone = get_backbone_network(type=cfg.MODEL.backbone_type)
        self.decoder = get_decoder_network(type=cfg.MODEL.backbone_type)

    def forward(self, inputs, mode='test'):
        image = inputs['input']['image'].to(self.device)

        if 'vit' in cfg.MODEL.backbone_type:
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        
        img_feat = self.backbone(image)
        contact_out, contact_336_out, contact_84_out, contact_joint_out = self.decoder(img_feat)
        
        return dict(contact_out=contact_out, contact_336_out=contact_336_out, contact_84_out=contact_84_out, contact_joint_out=contact_joint_out)



def get_backbone_network(type='hamer'):
    if type in ['hamer']:
        from lib.models.backbone.backbone_hamer_style import ViT_HaMeR
        backbone = ViT_HaMeR()
        checkpoint = torch.load(cfg.MODEL.hamer_backbone_pretrained_path, map_location='cuda')['state_dict']
        filtered_state_dict = {k[len("backbone."):]: v for k, v in checkpoint.items() if k.startswith("backbone.")}
        backbone.load_state_dict(filtered_state_dict)
    elif type in ['resnet-18']:
        from lib.models.backbone.resnet import ResNetBackbone
        backbone = ResNetBackbone(18) # ResNet
        backbone.init_weights()
    elif type in ['resnet-34']:
        from lib.models.backbone.resnet import ResNetBackbone
        backbone = ResNetBackbone(34) # ResNet
        backbone.init_weights()
    elif type in ['resnet-50']:
        from lib.models.backbone.resnet import ResNetBackbone
        backbone = ResNetBackbone(50) # ResNet
        backbone.init_weights()
    elif type in ['resnet-101']:
        from lib.models.backbone.resnet import ResNetBackbone
        backbone = ResNetBackbone(101) # ResNet
        backbone.init_weights()
    elif type in ['resnet-152']:
        from lib.models.backbone.resnet import ResNetBackbone
        backbone = ResNetBackbone(152) # ResNet
        backbone.init_weights()
    elif type in ['hrnet-w32']:
        from lib.models.backbone.hrnet import HighResolutionNet
        from lib.utils.func_utils import load_config
        config = load_config(cfg.MODEL.hrnet_w32_backbone_config_path)
        pretrained = cfg.MODEL.hrnet_w32_backbone_pretrained_path
        backbone = HighResolutionNet(config)
        backbone.init_weights(pretrained=pretrained)
    elif type in ['hrnet-w48']:
        from lib.models.backbone.hrnet import HighResolutionNet
        from lib.utils.func_utils import load_config
        config = load_config(cfg.MODEL.hrnet_w48_backbone_config_path)
        pretrained = cfg.MODEL.hrnet_w48_backbone_pretrained_path
        backbone = HighResolutionNet(config)
        backbone.init_weights(pretrained=pretrained)
    elif type in ['handoccnet']:
        from lib.models.backbone.fpn import FPN
        backbone = FPN(pretrained=False)
        pretrained = cfg.MODEL.handoccnet_backbone_pretrained_path
        state_dict = {k[len('module.backbone.'):]: v for k, v in torch.load(pretrained)['network'].items() if k.startswith('module.backbone.')}
        backbone.load_state_dict(state_dict, strict=True)
    elif type in ['vit-s-16']:
        from lib.models.backbone.vit import ViTBackbone
        backbone = ViTBackbone(model_name='vit_small_patch16_224', pretrained=True)
    elif type in ['vit-b-16']:
        from lib.models.backbone.vit import ViTBackbone
        backbone = ViTBackbone(model_name='vit_base_patch16_224', pretrained=True)
    elif type in ['vit-l-16']:
        from lib.models.backbone.vit import ViTBackbone
        backbone = ViTBackbone(model_name='vit_large_patch16_224', pretrained=True)
    else:
        raise NotImplementedError

    return backbone



def get_decoder_network(type='hamer'):
    from lib.models.decoder.decoder_hamer_style import ContactTransformerDecoderHead
    decoder = ContactTransformerDecoderHead()

    return decoder