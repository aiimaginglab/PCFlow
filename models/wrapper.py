import torch
import torch.nn as nn

from .taesd import TAESD


class DecoderFeatureExtractor(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z, return_features=False):
        feats = {}
        h = self.decoder.conv_in(z)

        # mid block (semantic)
        h = self.decoder.mid_block(h)
        feats["mid_block"] = h

        # upsampling blocks (texture/detail)
        for i, up_block in enumerate(self.decoder.up_blocks):
            h = up_block(h)
            feats[f"up_block_{i}"] = h

        # output convs (pixel-space)
        h = self.decoder.conv_norm_out(h)
        h = self.decoder.conv_act(h)
        out = self.decoder.conv_out(h)
        feats["conv_out"] = out

        return (out, feats) if return_features else out
    
class TAESDDecoderFeatureExtractor(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        # Define indices for feature extraction based on TAESD decoder structure
        # 0: conv, 1: ReLU
        # 2-4: Blocks
        # 5: Upsample, 6: conv (up_block_0)
        # 7-9: Blocks
        # 10: Upsample, 11: conv (up_block_1)
        # 12-14: Blocks
        # 15: Upsample, 16: conv (up_block_2)
        # 17: Block
        # 18: conv (out)

        self.feature_indices = {
            4: "mid_block",    # After first block sequence
            6: "up_block_0",   # After first upsample
            11: "up_block_1",  # After second upsample
            16: "up_block_2",  # After third upsample
            18: "conv_out"     # Final output
        }

    def forward(self, z, return_features=False):
        h = torch.tanh(z / 3) * 3
        
        feats = {}
        for i, layer in enumerate(self.decoder.layers):
            h = layer(h)
            if i in self.feature_indices:
                feats[self.feature_indices[i]] = h
        
        out = h
        return (out, feats) if return_features else out

# Wrapper for AutoencoderKL compatibility
class TAESDWrapper(nn.Module):
    def __init__(self, pretrained_model_name_or_path="madebyollin/taesd3"):
        super().__init__()
        self.model = TAESD(pretrained_model_name_or_path=pretrained_model_name_or_path)
        self.config = type('Config', (), {'scaling_factor': 1.0})() 

    def encode(self, x):
        latents = self.model.encoder(x)
        return DeterministicDistribution(latents)

    def decode(self, z):
        image = self.model.decoder(z)
        return DecoderOutput(image)

    def forward(self, x):
        return self.model(x)

class DeterministicDistribution:
    def __init__(self, latents):
        self.latents = latents
        self.latent_dist = self

    def sample(self, generator=None):
        return self.latents
    
    def mode(self):
        return self.latents

class DecoderOutput:
    def __init__(self, sample):
        self.sample = sample