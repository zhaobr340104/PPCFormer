"""Phase-window cell attention with a head-shared spatial--spectral relation bias."""

import math

import torch
from torch import nn


def normalize_values(values):
    values = torch.as_tensor(values, dtype=torch.float32)
    span = values.max() - values.min()
    return torch.zeros_like(values) if float(span) == 0 else (values - values.min()) / span


def phase_descriptors(config, wavelengths):
    phase = torch.arange(config.msfa_size ** 2, dtype=torch.long)
    denominator = max(config.msfa_size - 1, 1)
    return torch.stack((
        (phase // config.msfa_size).float() / denominator,
        (phase % config.msfa_size).float() / denominator,
        wavelengths[torch.as_tensor(config.pattern, dtype=torch.long)],
    ), dim=-1)


def build_relation_features(config, wavelengths):
    window, msfa = config.window_cells, config.msfa_size
    phases = msfa ** 2
    phase = torch.arange(phases, dtype=torch.long)
    cy, cx = torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")
    cy = cy.reshape(-1).repeat_interleave(phases)
    cx = cx.reshape(-1).repeat_interleave(phases)
    token_phase = phase.repeat(window ** 2)
    py, px = cy * msfa + token_phase // msfa, cx * msfa + token_phase % msfa
    band = torch.as_tensor(config.pattern, dtype=torch.long)
    token_lambda = wavelengths[band][token_phase]
    scale = float(max(msfa * window - 1, 1))
    dx = (px[None, :] - px[:, None]).float() / scale
    dy = (py[None, :] - py[:, None]).float() / scale
    dl = token_lambda[None, :] - token_lambda[:, None]
    return torch.stack((dx, dy, dl, dl.abs()), dim=-1)


class PWCA(nn.Module):
    def __init__(self, config, shift_cells=0):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.window_cells = config.window_cells
        self.phases = config.msfa_size ** 2
        self.shift_cells = shift_cells
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=True)
        self.output = nn.Linear(config.dim, config.dim)
        self.attention_dropout = nn.Dropout(0.0)
        self.output_dropout = nn.Dropout(0.0)
        wavelengths = normalize_values(config.wavelengths)
        self.register_buffer("phase_descriptors", phase_descriptors(config, wavelengths))
        self.register_buffer("relation_features", build_relation_features(config, wavelengths))
        self.relation_mlp = nn.Sequential(
            nn.Linear(4, config.relation_hidden),
            nn.GELU(),
            nn.Linear(config.relation_hidden, 1, bias=False),
        )
        self.relation_head_scale = nn.Parameter(torch.ones(config.num_heads))
        nn.init.zeros_(self.relation_mlp[-1].weight)
        self._window_metadata_cache = {}
        self._relation_bias_cache = {}
        self._attention_bias_cache = {}

    @staticmethod
    def _cache_key_device(device):
        return str(torch.device(device))

    @staticmethod
    def _cache_put(cache, key, value, max_items=16):
        if len(cache) >= max_items:
            cache.clear()
        cache[key] = value
        return value

    @staticmethod
    def _tensor_cache_version(tensor):
        # Include identity as well as in-place updates (e.g. replaced buffers).
        # Inference tensors have no version counter, so cannot be cached safely.
        if torch.is_inference(tensor):
            return None
        return id(tensor), int(tensor._version)

    @staticmethod
    def _parameter_versions(module):
        if module is None:
            return ()
        versions = tuple(
            PWCA._tensor_cache_version(parameter) for parameter in module.parameters()
        )
        return None if None in versions else versions

    def _relation_parameter_versions(self):
        versions = self._parameter_versions(self.relation_mlp)
        if versions is None:
            return None
        if self.relation_head_scale is None:
            return versions
        scale_version = self._tensor_cache_version(self.relation_head_scale)
        if scale_version is None:
            return None
        return versions + (scale_version,)

    def _clear_parameter_dependent_cache(self):
        self._relation_bias_cache.clear()
        self._attention_bias_cache.clear()

    def _clear_cache(self):
        self._window_metadata_cache.clear()
        self._clear_parameter_dependent_cache()

    def _apply(self, fn, *args, **kwargs):
        # Cached tensors are ordinary attributes, so Module.to() cannot move
        # them. Also discard results computed before a dtype round trip.
        self._clear_cache()
        return super()._apply(fn, *args, **kwargs)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        self._clear_cache()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._clear_cache()
        return self

    @staticmethod
    def _partition_cell_map(cell_map, window_cells):
        height, width = cell_map.shape
        return (
            cell_map.reshape(
                height // window_cells,
                window_cells,
                width // window_cells,
                window_cells,
            )
            .permute(0, 2, 1, 3)
            .reshape(-1, window_cells ** 2)
        )

    def _window_metadata(self, height, width, device):
        window = self.window_cells
        padded_height = int(math.ceil(float(height) / window) * window)
        padded_width = int(math.ceil(float(width) / window) * window)

        valid = torch.zeros(padded_height, padded_width, dtype=torch.bool, device=device)
        valid[:height, :width] = True
        cell_ids = torch.full(
            (padded_height, padded_width), -1, dtype=torch.long, device=device
        )
        cell_ids[:height, :width] = torch.arange(
            height * width, device=device
        ).reshape(height, width)

        region = torch.zeros(
            padded_height, padded_width, dtype=torch.long, device=device
        )
        if self.shift_cells:
            shift = self.shift_cells
            height_slices = (
                slice(0, -window),
                slice(-window, -shift),
                slice(-shift, None),
            )
            width_slices = (
                slice(0, -window),
                slice(-window, -shift),
                slice(-shift, None),
            )
            region_id = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    region[height_slice, width_slice] = region_id
                    region_id += 1
            valid = torch.roll(valid, shifts=(-shift, -shift), dims=(0, 1))
            cell_ids = torch.roll(cell_ids, shifts=(-shift, -shift), dims=(0, 1))

        valid_cells = self._partition_cell_map(valid, window)
        id_cells = self._partition_cell_map(cell_ids, window)
        region_cells = self._partition_cell_map(region, window)
        valid_tokens = valid_cells.repeat_interleave(self.phases, dim=1)
        id_tokens = id_cells.repeat_interleave(self.phases, dim=1)
        region_tokens = region_cells.repeat_interleave(self.phases, dim=1)

        forbidden = region_tokens[:, :, None] != region_tokens[:, None, :]
        forbidden = forbidden | (~valid_tokens[:, None, :])
        invalid_queries = (~valid_tokens)[:, :, None]
        forbidden = forbidden | invalid_queries
        diagonal = torch.eye(
            forbidden.shape[-1], dtype=torch.bool, device=device
        ).unsqueeze(0)
        forbidden = forbidden & ~(invalid_queries & diagonal)
        return padded_height, padded_width, id_tokens, valid_tokens, forbidden

    def window_metadata(self, height, width, device=None):
        if device is None:
            device = self.phase_descriptors.device
        key = (
            int(height), int(width), self.window_cells, self.shift_cells,
            self.phases, self._cache_key_device(device),
        )
        cached = self._window_metadata_cache.get(key)
        if cached is not None:
            return cached
        return self._cache_put(
            self._window_metadata_cache,
            key,
            self._window_metadata(height, width, device),
        )

    def _cached_relation_bias(self, dtype):
        if self.relation_mlp is None:
            return None
        use_cache = (
            (not self.training) and (not torch.is_grad_enabled())
            and not torch.is_inference(self.relation_features)
        )
        versions = self._relation_parameter_versions() if use_cache else None
        use_cache = use_cache and versions is not None
        key = None
        if use_cache:
            key = (
                self._cache_key_device(self.relation_features.device),
                str(dtype),
                versions,
                self._tensor_cache_version(self.relation_features),
            )
            cached = self._relation_bias_cache.get(key)
            if cached is not None:
                return cached

        relation_bias = self.relation_mlp(self.relation_features)
        if self.relation_head_scale is not None:
            relation_bias = (
                relation_bias[..., 0].unsqueeze(0)
                * self.relation_head_scale[:, None, None]
            )
        else:
            relation_bias = relation_bias.permute(2, 0, 1)
        relation_bias = relation_bias.to(dtype=dtype)
        if use_cache:
            return self._cache_put(self._relation_bias_cache, key, relation_bias)
        return relation_bias

    def _cached_attention_bias(self, forbidden, dtype, window_geometry):
        use_cache = (
            (not self.training) and (not torch.is_grad_enabled())
            and not torch.is_inference(self.relation_features)
        )
        if not use_cache:
            return None
        attention_bias_elements = (
            int(forbidden.shape[0])
            * self.num_heads
            * int(forbidden.shape[1])
            * int(forbidden.shape[2])
        )
        if attention_bias_elements > 8_000_000:
            return None
        versions = self._relation_parameter_versions()
        if versions is None:
            return None
        key = (
            window_geometry,
            int(forbidden.shape[0]),
            int(forbidden.shape[1]),
            self._cache_key_device(forbidden.device),
            str(dtype),
            versions,
            self._tensor_cache_version(self.relation_features),
        )
        cached = self._attention_bias_cache.get(key)
        if cached is not None:
            return cached

        mask_value = torch.finfo(dtype).min
        if self.relation_mlp is None:
            attention_bias = torch.zeros(
                forbidden.shape[0],
                self.num_heads,
                forbidden.shape[1],
                forbidden.shape[2],
                device=forbidden.device,
                dtype=dtype,
            )
        else:
            relation_bias = self._cached_relation_bias(dtype)
            attention_bias = relation_bias.unsqueeze(0).expand(
                forbidden.shape[0], -1, -1, -1
            ).clone()
        attention_bias = attention_bias.masked_fill(forbidden[:, None], mask_value)
        return self._cache_put(self._attention_bias_cache, key, attention_bias)

    def _partition_windows(self, x):
        batch, height, width, phases, dim = x.shape
        window = self.window_cells
        return (
            x.reshape(
                batch,
                height // window,
                window,
                width // window,
                window,
                phases,
                dim,
            )
            .permute(0, 1, 3, 2, 4, 5, 6)
            .reshape(-1, window * window * phases, dim)
        )

    def _reverse_windows(self, windows, batch, height, width):
        window = self.window_cells
        phases = self.phases
        return (
            windows.reshape(
                batch,
                height // window,
                width // window,
                window,
                window,
                phases,
                self.dim,
            )
            .permute(0, 1, 3, 2, 4, 5, 6)
            .reshape(batch, height, width, phases, self.dim)
        )

    def forward(self, x):
        batch, height, width, phases, dim = x.shape
        if phases != self.phases or dim != self.dim:
            raise ValueError("PWCA input has an incompatible phase or feature dimension")
        padded_height, padded_width, _, _, forbidden = self.window_metadata(
            height, width, x.device
        )
        if padded_height == height and padded_width == width:
            padded = x
        else:
            padded = x.new_zeros(batch, padded_height, padded_width, phases, dim)
            padded[:, :height, :width] = x
        if self.shift_cells:
            padded = torch.roll(
                padded,
                shifts=(-self.shift_cells, -self.shift_cells),
                dims=(1, 2),
            )

        windows = self._partition_windows(padded)
        tokens = windows.shape[1]
        qkv = self.qkv(windows).reshape(
            windows.shape[0], tokens, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = (q @ k.transpose(-2, -1)) * self.scale

        num_windows = forbidden.shape[0]
        needs_window_mask = (
            self.shift_cells
            or padded_height != height
            or padded_width != width
        )
        if not needs_window_mask:
            relation_bias = self._cached_relation_bias(logits.dtype)
            if relation_bias is not None:
                logits = logits + relation_bias.unsqueeze(0)
            attention = self.attention_dropout(logits.softmax(dim=-1))
        else:
            logits = logits.reshape(batch, num_windows, self.num_heads, tokens, tokens)
            # Window count/token count alone do not identify the mask: a
            # 4x8 cell grid and an 8x4 grid have different shifted boundaries.
            window_geometry = (
                int(height), int(width), int(padded_height), int(padded_width),
                self.window_cells, self.shift_cells, self.phases,
            )
            attention_bias = self._cached_attention_bias(
                forbidden, logits.dtype, window_geometry
            )
            if attention_bias is not None:
                logits = logits + attention_bias.unsqueeze(0)
            else:
                relation_bias = self._cached_relation_bias(logits.dtype)
                if relation_bias is not None:
                    logits = logits + relation_bias.unsqueeze(0).unsqueeze(0)
                logits = logits.masked_fill(
                    forbidden[None, :, None], torch.finfo(logits.dtype).min
                )
            attention = self.attention_dropout(logits.softmax(dim=-1))
        attention = attention.reshape(-1, self.num_heads, tokens, tokens)
        output = attention @ v
        output = output.transpose(1, 2).reshape(windows.shape[0], tokens, dim)
        output = self.output_dropout(self.output(output))
        output = self._reverse_windows(output, batch, padded_height, padded_width)
        if self.shift_cells:
            output = torch.roll(
                output,
                shifts=(self.shift_cells, self.shift_cells),
                dims=(1, 2),
            )
        return output[:, :height, :width]
