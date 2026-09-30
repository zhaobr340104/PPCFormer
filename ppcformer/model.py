"""PPCFormer architecture, cell-domain reconstruction, and MSFA operations."""

from dataclasses import dataclass, field
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .attention import PWCA, normalize_values, phase_descriptors

PATTERN = tuple(range(16))


@dataclass(frozen=True)
class ModelConfig:
    wavelengths: tuple
    pattern: tuple = PATTERN
    msfa_size: int = field(default=4, init=False)
    num_bands: int = field(default=16, init=False)
    dim: int = field(default=48, init=False)
    num_blocks: int = field(default=6, init=False)
    num_heads: int = field(default=6, init=False)
    window_cells: int = field(default=2, init=False)
    mlp_ratio: int = field(default=2, init=False)
    relation_hidden: int = field(default=64, init=False)
    mosaic_stem_hidden: int = field(default=48, init=False)
    cell_context_hidden: int = field(default=64, init=False)
    cell_context_dilations: tuple = field(default=(1, 2, 3, 4, 5, 6), init=False)
    synthesis_hidden: int = field(default=96, init=False)
    synthesis_dilations: tuple = field(default=(1, 2, 3), init=False)
    measurement_refinement_hidden: int = field(default=48, init=False)
    measurement_refinement_dilations: tuple = field(default=(1, 2, 4, 6), init=False)

    def __post_init__(self):
        if len(self.wavelengths) != 16 or not all(math.isfinite(x) for x in self.wavelengths):
            raise ValueError("Exactly 16 finite wavelengths are required, in cube-channel order")
        if len(set(self.wavelengths)) != 16:
            raise ValueError("Wavelengths must be distinct")
        if sorted(self.pattern) != list(range(16)):
            raise ValueError("pattern must map the 16 row-major phases to 16 distinct bands")


def get_wb_filter_msfa(msfa_size):
    """Return the fixed bilinear kernel used by 4x4 MSFA WB interpolation."""
    size = 2 * msfa_size - 1
    line = []
    column = []
    for index in range(size):
        if (index + 1) <= np.floor(math.sqrt(msfa_size ** 2)):
            line.append(index + 1)
            column.append(index + 1)
        else:
            line.append(line[index - 1] - 1.0)
            column.append(column[index - 1] - 1.0)

    bilinear_filter = np.zeros(size * size)
    for row in range(size):
        for col in range(size):
            bilinear_filter[col + row * size] = (
                line[row] * column[col] / (msfa_size ** 2)
            )
    return torch.from_numpy(bilinear_filter.reshape(size, size)).float()


def phase_safe_pad_msfa(x, msfa_size, pad_width, pad_height):
    """Right/bottom pad without changing the spatial phase of MSFA samples."""
    if pad_width == 0 and pad_height == 0:
        return x
    height, width = x.shape[-2:]
    target_height = height + pad_height
    target_width = width + pad_width
    if target_height % msfa_size or target_width % msfa_size:
        raise ValueError("phase-safe padding target must contain complete MSFA cells")
    if height < msfa_size or width < msfa_size:
        raise ValueError("input must contain at least one complete MSFA cell")

    padded = x.new_empty(*x.shape[:-2], target_height, target_width)
    target_cell_height = target_height // msfa_size
    target_cell_width = target_width // msfa_size
    for phase_y in range(msfa_size):
        for phase_x in range(msfa_size):
            phase_plane = x[..., phase_y::msfa_size, phase_x::msfa_size]
            phase_pad_height = target_cell_height - phase_plane.shape[-2]
            phase_pad_width = target_cell_width - phase_plane.shape[-1]
            reflect_safe = (
                (phase_pad_width == 0 or phase_pad_width < phase_plane.shape[-1])
                and (
                    phase_pad_height == 0
                    or phase_pad_height < phase_plane.shape[-2]
                )
            )
            mode = "reflect" if reflect_safe else "replicate"
            padded_phase = F.pad(
                phase_plane,
                (0, phase_pad_width, 0, phase_pad_height),
                mode=mode,
            )
            padded[..., phase_y::msfa_size, phase_x::msfa_size] = padded_phase
    return padded


class PhaseLocalMixer(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, height, width, phases, dim = x.shape
        y = x.permute(0, 3, 4, 1, 2).reshape(batch * phases, dim, height, width)
        y = self.depthwise(y)
        y = F.gelu(y)
        y = self.pointwise(y)
        y = self.dropout(y)
        return y.reshape(batch, phases, dim, height, width).permute(0, 3, 4, 1, 2)


class TokenMLPFFN(nn.Module):
    """Shared two-layer MLP applied independently to each cell token."""

    def __init__(self, dim, mlp_ratio=2, dropout=0.0):
        super().__init__()
        hidden_dim = dim * mlp_ratio
        self.expand = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = F.gelu(self.expand(x))
        return self.dropout(self.project(y))


class CellSynthesisBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0.0):
        super().__init__()
        dilation = int(dilation)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            1,
            dilation,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = self.depthwise(x)
        residual = F.gelu(residual)
        residual = self.pointwise(residual)
        return x + self.dropout(residual)


class CellDomainSynthesisHead(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels,
                 dilations, dropout=0.0):
        super().__init__()
        self.reduce = nn.Conv2d(input_channels, hidden_channels, 1)
        self.blocks = nn.Sequential(
            *(
                CellSynthesisBlock(hidden_channels, dilation, dropout)
                for dilation in dilations
            )
        )
        self.project = nn.Conv2d(hidden_channels, output_channels, 1)
        self.zero_initialize_output()

    def zero_initialize_output(self):
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        x = F.gelu(self.reduce(x))
        x = self.blocks(x)
        return self.project(x)


class CellContextBlock(nn.Module):
    def __init__(self, channels, dilation=1, expansion=2, dropout=0.0):
        super().__init__()
        dilation = int(dilation)
        hidden_channels = channels * expansion
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            3,
            1,
            dilation,
            dilation=dilation,
            groups=channels,
        )
        self.expand = nn.Conv2d(channels, hidden_channels, 1)
        self.project = nn.Conv2d(hidden_channels, channels, 1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = self.depthwise(x)
        residual = F.gelu(residual)
        residual = F.gelu(self.expand(residual))
        residual = self.project(residual)
        return x + self.dropout(residual)


class DilatedCellContext(nn.Module):
    """Residual dilated context propagation on the MSFA-cell lattice."""

    def __init__(self, channels, hidden_channels, dilations, dropout=0.0):
        super().__init__()
        self.reduce = nn.Conv2d(channels, hidden_channels, 1)
        self.blocks = nn.Sequential(
            *(
                CellContextBlock(hidden_channels, dilation, dropout=dropout)
                for dilation in dilations
            )
        )
        self.project = nn.Conv2d(hidden_channels, channels, 1)
        self.zero_initialize_output()

    def zero_initialize_output(self):
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        y = F.gelu(self.reduce(x))
        y = self.blocks(y)
        return x + self.project(y)


class MosaicContext(nn.Module):
    """Native-grid context mapped to phase-aligned cell tokens."""

    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.msfa_size = config.msfa_size
        hidden = config.mosaic_stem_hidden
        self.stem = nn.Conv2d(1, hidden, 3, 1, 1)
        self.blocks = nn.Sequential(CellContextBlock(hidden, 1), CellContextBlock(hidden, 2))
        self.project = nn.Conv2d(hidden, 16 * config.dim, 4, stride=4)
        self.zero_initialize_output()

    def zero_initialize_output(self):
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, raw):
        features = self.blocks(F.gelu(self.stem(raw)))
        bias = self.project(features)
        batch, _, height, width = bias.shape
        return bias.reshape(batch, 16, self.dim, height, width).permute(0, 3, 4, 1, 2)


class MeasurementRefinement(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.measurement_refinement_hidden
        self.stem = nn.Conv2d(48, hidden, 1)
        self.blocks = nn.Sequential(*(
            CellContextBlock(hidden, dilation) for dilation in config.measurement_refinement_dilations
        ))
        self.project = nn.Conv2d(hidden, 16, 1)
        self.zero_initialize_output()

    def zero_initialize_output(self):
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, prediction, sparse_raw, mask):
        mask = mask.expand(prediction.shape[0], -1, -1, -1).to(dtype=prediction.dtype)
        observed_residual = sparse_raw - prediction * mask
        features = torch.cat((prediction, observed_residual, mask), dim=1)
        return prediction + self.project(self.blocks(F.gelu(self.stem(features))))


class TransformerBlock(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.local_norm = nn.LayerNorm(config.dim)
        self.local_mixer = PhaseLocalMixer(config.dim)
        self.attention_norm = nn.LayerNorm(config.dim)
        self.attention = PWCA(config, shift_cells=0 if index % 2 == 0 else 1)
        self.ffn_norm = nn.LayerNorm(config.dim)
        self.ffn = TokenMLPFFN(config.dim, config.mlp_ratio)

    def forward(self, x):
        x = x + self.local_mixer(self.local_norm(x))
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class PPCFormer(nn.Module):
    """Reconstruct [N,16,H,W] from a [N,1,H,W] raw mosaic.

    wavelengths are in cube-channel order, not necessarily wavelength order.
    pattern maps each row-major 4x4 filter phase to a cube-channel index.
    An optional precomputed sparse_raw avoids rebuilding the sparse observation.
    """

    def __init__(self, wavelengths, pattern=PATTERN):
        super().__init__()
        self.config = config = ModelConfig(tuple(wavelengths), tuple(pattern))
        self.msfa_size = 4
        self.num_bands = self.phases = 16
        self.register_buffer("phase_to_band", torch.tensor(config.pattern, dtype=torch.long))
        self.register_buffer("pattern", torch.tensor(config.pattern, dtype=torch.long).reshape(4, 4))
        self.register_buffer("phase_descriptors", phase_descriptors(config, normalize_values(config.wavelengths)))
        self.WB_Conv = nn.Conv2d(16, 16, 7, padding=3, groups=16, bias=False)
        kernel = get_wb_filter_msfa(4).reshape(1, 1, 7, 7).repeat(16, 1, 1, 1)
        self.WB_Conv.weight = nn.Parameter(kernel, requires_grad=False)
        self.token_embedding = nn.Linear(1, config.dim)
        self.mosaic_stem_prior = MosaicContext(config)
        self.blocks = nn.ModuleList(TransformerBlock(config, i) for i in range(6))
        self.trunk_norm = nn.LayerNorm(config.dim)
        self.trunk_projection = nn.Linear(config.dim, config.dim)
        # Preserve construction order as well as parameter names.
        self.reconstruction_head = CellDomainSynthesisHead(768, 256, 96, (1, 2, 3))
        self.cell_context = DilatedCellContext(768, 64, (1, 2, 3, 4, 5, 6))
        self.measurement_refinement = MeasurementRefinement(config)
        self._measurement_mask_cache = {}

    def _apply(self, fn, *args, **kwargs):
        self._measurement_mask_cache.clear()
        return super()._apply(fn, *args, **kwargs)

    def train(self, mode=True):
        # Validation may have created inference-mode tensors. Do not reuse
        # those constants in a subsequent autograd graph.
        if mode:
            self._measurement_mask_cache.clear()
        return super().train(mode)

    def measurement_mask(self, height, width, device, dtype):
        key = (height, width, str(device), str(dtype))
        if key not in self._measurement_mask_cache:
            if len(self._measurement_mask_cache) >= 16:
                self._measurement_mask_cache.clear()
            mask = torch.zeros(1, 16, height, width, device=device, dtype=dtype)
            for phase, band in enumerate(self.config.pattern):
                mask[:, band, phase // 4::4, phase % 4::4] = 1
            self._measurement_mask_cache[key] = mask
        return self._measurement_mask_cache[key]

    def forward(self, raw, sparse_raw=None):
        if raw.ndim != 4 or raw.shape[1] != 1 or not raw.is_floating_point():
            raise ValueError("raw must be a floating tensor with shape [N,1,H,W]")
        height, width = raw.shape[-2:]
        if min(height, width) < 4:
            raise ValueError("Each spatial dimension must contain at least four pixels")
        mask = self.measurement_mask(height, width, raw.device, raw.dtype)
        if sparse_raw is None:
            sparse_raw = raw * mask
        if sparse_raw.shape != (raw.shape[0], 16, height, width):
            raise ValueError("sparse_raw must have shape [N,16,H,W]")
        if sparse_raw.device != raw.device or sparse_raw.dtype != raw.dtype:
            raise ValueError("raw and sparse_raw must have the same device and dtype")
        pad_h, pad_w = (-height) % 8, (-width) % 8
        raw_padded = phase_safe_pad_msfa(raw, 4, pad_w, pad_h)
        sparse_padded = phase_safe_pad_msfa(sparse_raw, 4, pad_w, pad_h)
        cells = F.pixel_unshuffle(raw_padded, 4).permute(0, 2, 3, 1)
        stem = self.token_embedding(cells.unsqueeze(-1)) + self.mosaic_stem_prior(raw_padded)
        features = stem
        for block in self.blocks:
            features = block(features)
        features = stem + self.trunk_projection(self.trunk_norm(features))
        batch, hc, wc, phases, dim = features.shape
        packed = features.permute(0, 3, 4, 1, 2).reshape(batch, phases * dim, hc, wc)
        residual = F.pixel_shuffle(self.reconstruction_head(self.cell_context(packed)), 4)
        initial = (self.WB_Conv(sparse_padded) + residual)[:, :, :height, :width]
        refined = self.measurement_refinement(initial, sparse_raw, mask)
        return refined * (1.0 - mask) + sparse_raw
