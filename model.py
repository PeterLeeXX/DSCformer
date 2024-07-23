import torch
import torch.nn as nn
from functools import partial
from einops.layers.torch import Rearrange
from torchsummary import summary


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """""
    This section is model pruning, which is not used in the default model of the thesis.
    """""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, input_shape=4200, patch_size=60, num_features=60, in_chans=1, norm_layer=None,
                 flatten=True):
        super().__init__()
        self.flatten = flatten
        self.proj = nn.Conv1d(in_chans, num_features, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        """""
         The spectral data of (B,1,4200) is split patch by convolutional layer to get the output of (B,num_features, patch_size)
        """""
        if self.flatten:
            x = x.transpose(1, 2)
            #   transpose matrix，(B,C,L) ——> (B,N,C)
        x = self.norm(x)
        return x


class DeepWiseSeparableConv(nn.Module):
    def __init__(self, ch_in):
        super(DeepWiseSeparableConv, self).__init__()
        self.depth_conv = nn.Conv1d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv1d(ch_in, ch_in, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class TokenMixer(nn.Module):
    def __init__(self, ch_in, mixer_param):
        """""
        The TokenMixer function passes two parameters, ch_in is the number of channels (dimensions) of the incoming 
        tensor and mixer_param is the kind of TokenMixer to choose from in this section, optionally [dsc, conv, mlp, 
        pool].
        """""
        super().__init__()
        self.ch_in = ch_in
        self.token_mixer = mixer_param
        self.dsc = DeepWiseSeparableConv(ch_in)

        self.conv = nn.Conv1d(ch_in, ch_in, kernel_size=3, padding=1)

        self.mlp = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(ch_in, ch_in),
            nn.GELU(),
            nn.Linear(ch_in, ch_in),
            Rearrange('b d n -> b n d'))

        self.pool = nn.AvgPool1d(3, 1, padding=1)

        mixer_dict = {
            'dsc': self._init_dsc,
            'conv': self._init_conv,
            'mlp': self._init_mlp,
            'pool': self._init_pool
        }
        if self.token_mixer in mixer_dict:
            mixer_dict[self.token_mixer]()
        else:
            raise ValueError(f"Unknown parameter: {self.token_mixer}")

    def _init_dsc(self):
        self.layers = self.dsc

    def _init_conv(self):
        self.layers = self.conv

    def _init_mlp(self):
        self.layers = self.mlp

    def _init_pool(self):
        self.layers = self.pool

    def forward(self, x):
        x = self.layers(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., attn_drop=0., token_mixer='dsc',
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = TokenMixer(ch_in=(4200//dim), mixer_param=token_mixer)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DSCformer(nn.Module):
    def __init__(
            self, input_shape=4200, patch_size=60, in_chans=1, num_classes=1, num_features=60, token_mixer='dsc',
            depth=1, mlp_ratio=1., p_divide=1, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-5), act_layer=nn.GELU):

        super().__init__()

        self.patch_embed = PatchEmbed(input_shape=input_shape, patch_size=patch_size, in_chans=in_chans,
                                      num_features=num_features)
        num_patches = 4200 // patch_size
        self.num_features = num_features

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, num_features))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=num_features,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    token_mixer=token_mixer,
                    norm_layer=norm_layer,
                    act_layer=act_layer
                ) for i in range(depth)
            ]
        )
        self.p_divide = p_divide
        self.norm = norm_layer(num_features)
        self.pool = nn.MaxPool1d(kernel_size=patch_size//self.p_divide, stride=patch_size//self.p_divide)
        self.head = nn.Linear(4200//num_features*p_divide, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pool(x)
        if self.p_divide == 1:
            x = x.squeeze(-1)
        else:
            x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

    def freeze_backbone(self):
        backbone = [self.patch_embed, self.cls_token, self.pos_embed, self.pos_drop, self.blocks[:8]]
        for module in backbone:
            try:
                for param in module.parameters():
                    param.requires_grad = False
            except:
                module.requires_grad = False

    def unfreeze_backbone(self):
        backbone = [self.patch_embed, self.cls_token, self.pos_embed, self.pos_drop, self.blocks[:8]]
        for module in backbone:
            try:
                for param in module.parameters():
                    param.requires_grad = True
            except:
                module.requires_grad = True


def dsc(input_shape=4200, patch_size=60, depth=1, p_divide=1, mlp_ratio=1., token_mixer='dsc'):
    model = DSCformer(input_shape, patch_size=patch_size, num_features=patch_size, depth=depth, p_divide=p_divide,
                      mlp_ratio=mlp_ratio, token_mixer=token_mixer)
    return model


# 以下为测试模型用代码
def main():
    input = torch.randn(1, 1, 4200)
    model = dsc()
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)
    summary(model=model, input_size=(1, 4200), batch_size=1, device="cpu")


if __name__ == "__main__":
    main()
