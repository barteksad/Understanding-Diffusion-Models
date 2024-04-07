import math
from functools import partial
from typing import Callable, Optional, TypeVar, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import einsum

T = TypeVar("T")


def exists(x: Optional[T]) -> bool:
    return x is not None


def default(val: Optional[T], d: Union[T, Callable[[], T]]) -> T:
    if exists(val):
        return val  # type: ignore
    else:
        return d() if callable(d) else d


def num2groups(num, divisor):
    groups = [divisor] * (num // divisor)
    if num % divisor != 0:
        groups.append(num % divisor)
    return groups


class ResBlock(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.fn(x, *args, **kwargs)


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


def Upsample(dim: int, out_dim: Optional[int]) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(out_dim, dim), 3, padding=1),
    )


class SinusoidalPositionlEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class WeightedStandarizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o c h w -> o 1 1 1", reduction="mean")
        var = reduce(
            weight, "o c h w -> o 1 1 1", reduction=partial(torch.var, unbiased=False)
        )
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, groups: int = 8):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.proj = WeightedStandarizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: Optional[torch.Tensor] = None,
        z_sem: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift  # type: ignore
            if exists(z_sem):
                x = z_sem * (scale * x + shift)
            else:
                x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        time_emb_dim: Optional[int] = None,
        z_sem_dim: Optional[int] = None,
        groups: int = 8,
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))  # type: ignore
            )
            if exists(time_emb_dim)
            else None
        )
        self.z_sem_mlp = (
            nn.Sequential(
                nn.Sequential(nn.SiLU(), nn.Linear(z_sem_dim, dim_out))  # type: ignore
            )
            if exists(z_sem_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        z_sem: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale_shift = None
        if exists(t) and exists(self.mlp):
            time_emb = self.mlp(t)  # type: ignore
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        if exists(z_sem) and exists(self.z_sem_mlp):
            z_sem = self.z_sem_mlp(z_sem)  # type: ignore
            z_sem = rearrange(z_sem, "b c -> b c 1 1")

        h = self.block1(x, scale_shift, z_sem)
        h = self.block2(h)
        return self.res_conv(x) + h


class Attention(nn.Module):
    def __init__(self, dim: int, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = einsum("b h d n, b h e n -> b h d e", k, v)

        out = einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class Encoder(nn.Module):

    def __init__(
        self,
        in_out: list[Tuple[int, int]],
        time_dim: Optional[int],
        num_resolutions: int,
        block_class: partial[ResnetBlock],
    ):
        super().__init__()
        self.downs = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_class(dim_in, dim_in, time_dim),
                        block_class(dim_in, dim_in, time_dim),
                        ResBlock(PreNorm(dim_in, LinearAttention(dim_in))),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, z_sem: torch.Tensor
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        h = []

        for down1, down2, attn, downsample in self.downs:  # type: ignore
            x = down1(x, t, z_sem)
            h.append(x)

            x = down2(x, t, z_sem)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        return x, h


class MidBlock(nn.Module):
    def __init__(
        self, dim: int, time_dim: Optional[int], z_sem_dim: Optional[int] = None
    ):
        super().__init__()
        self.block1 = ResnetBlock(dim, dim, time_dim, z_sem_dim)
        self.attn = ResBlock(PreNorm(dim, Attention(dim)))
        self.block2 = ResnetBlock(dim, dim, time_dim, z_sem_dim)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, z_sem: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = self.block1(x, t, z_sem)
        x = self.attn(x)
        x = self.block2(x, t, z_sem)
        return x


class Decoder(nn.Module):

    def __init__(
        self,
        in_out: list[Tuple[int, int]],
        time_dim: Optional[int],
        block_class: partial[ResnetBlock],
    ):
        super().__init__()
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_class(dim_out + dim_in, dim_out, time_dim),
                        block_class(dim_out + dim_in, dim_out, time_dim),
                        ResBlock(PreNorm(dim_out, LinearAttention(dim_out))),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        h: list[torch.Tensor],
        t: torch.Tensor,
        z_sem: torch.Tensor,
    ) -> torch.Tensor:

        for up1, up2, attn, upsample in self.ups:  # type: ignore
            x = torch.cat((x, h.pop()), dim=1)
            x = up1(x, t, z_sem)

            x = torch.cat((x, h.pop()), dim=1)
            x = up2(x, t, z_sem)
            x = attn(x)

            x = upsample(x)

        return x


class DiffAE(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: Optional[int],
        out_dim: int,
        dim_mults: Tuple[int, ...],
        channels: int,
        resnet_block_groups: int,
    ):
        super().__init__()

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda d: dim * d, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_class = partial(ResnetBlock, groups=resnet_block_groups)

        num_resolutions = len(in_out)
        mid_dim = dims[-1]

        self.encoder = Encoder(in_out, None, num_resolutions, block_class)
        self.mid_block = MidBlock(mid_dim, None)

        self.output_proj = nn.Sequential(
            WeightedStandarizedConv2d(mid_dim, out_dim, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            Rearrange("b c 1 1 -> b c"),
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = self.init_conv(x)

        x, _ = self.encoder(x, None, None)
        x = self.mid_block(x, None, None)
        x = self.output_proj(x)

        return x


class Unet(nn.Module):

    def __init__(
        self,
        dim: int,
        init_dim: Optional[int],
        out_dim: Optional[int],
        dim_mults: Tuple[int, ...],
        channels: int,
        self_condition: bool,
        resnet_block_groups: int,
        time_dim_mult: int,
        z_sem_dim: int,
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda d: dim * d, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_class = partial(
            ResnetBlock, groups=resnet_block_groups, z_sem_dim=z_sem_dim
        )

        time_dim = dim * time_dim_mult

        self.time_mlp = nn.Sequential(
            SinusoidalPositionlEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        num_resolutions = len(in_out)
        mid_dim = dims[-1]

        self.encoder = Encoder(in_out, time_dim, num_resolutions, block_class)
        self.mid_block = MidBlock(mid_dim, time_dim, z_sem_dim)
        self.decoder = Decoder(in_out, time_dim, block_class)

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_class(dim * 2, dim, time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        # with torch.no_grad():
        #     for param in self.final_conv.parameters():
        #         param.zero_()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z_sem: Optional[torch.Tensor] = None,
        x_self_cond: Optional[torch.Tensor] = None,
    ):
        if self.self_condition:
            if not exists(x_self_cond):
                x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, x_self_cond), dim=1)  # type: ignore

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(t)

        x, h = self.encoder(x, t, z_sem)

        x = self.mid_block(x, t, z_sem)

        x = self.decoder(x, h, t, z_sem)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t, z_sem)
        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    diff_ae = DiffAE(
        dim=32,
        init_dim=None,
        out_dim=256,
        dim_mults=(1, 2, 4),
        channels=1,
        resnet_block_groups=4,
    )
    unet = Unet(
        dim=32,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4),
        channels=1,
        self_condition=False,
        resnet_block_groups=4,
        time_dim_mult=8,
        z_sem_dim=256,
    )
    x = torch.randn(2, 1, 32, 32)
    t = torch.tensor([1, 2], dtype=torch.long)
    z_sem = diff_ae(x)
    print(z_sem.shape)
    x_t = unet(x, t, z_sem)
    print(x_t.shape)
