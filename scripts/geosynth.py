import einops
import torch
import torch as th
import torch.nn as nn

from ControlNet.ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ControlNet.ldm.modules.attention import SpatialTransformer
from ControlNet.ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock,
)
from ControlNet.ldm.models.diffusion.ddpm import LatentDiffusion
from ControlNet.ldm.util import log_txt_as_img, exists, instantiate_from_config
from ControlNet.ldm.models.diffusion.ddim import DDIMSampler


class LocationEncoder(nn.Module):
    def __init__(self, embed_dim=256, out_dim=256, num_heads=4):
        super().__init__()
        self.query_embed = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_embed = nn.Linear(1280, embed_dim, bias=False)
        self.value_embed = nn.Linear(1280, embed_dim, bias=False)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ff_zero = zero_module(nn.Linear(embed_dim, out_dim, bias=False))

    def forward(self, loc_emb, time_emb):
        q, k, v = (
            self.query_embed(loc_emb),
            self.key_embed(time_emb),
            self.value_embed(time_emb),
        )
        x, _ = self.cross_attn(q, k, v)
        x = self.norm1(x) + loc_emb
        x = self.norm2(self.ff(x)) + x
        return x, self.ff_zero(x)


# class ControlledUnetModel(UNetModel):
#     def forward(
#         self,
#         x,
#         timesteps=None,
#         context=None,
#         control=None,
#         location=None,
#         only_mid_control=False,
#         **kwargs,
#     ):
#         print(f"\n--- ControlledUnetModel Forward ---")
#         print(f"[ControlledUnet] Input x shape: {x.shape}")
#         if timesteps is not None: print(f"[ControlledUnet] timesteps shape: {timesteps.shape}")
#         if context is not None: print(f"[ControlledUnet] context (text_emb) shape: {context.shape}")
#         if control is not None: print(f"[ControlledUnet] Received control signals: {len(control)} tensors. Shapes: {[c.shape for c in control]}")
#         if location is not None: print(f"[ControlledUnet] Received location signals: {len(location)} tensors. Shapes: {[l.shape for l in location]}")

#         hs = []
#         # import code; code.interact(local=locals());
#         with torch.no_grad():
#             t_emb = timestep_embedding(
#                 timesteps, self.model_channels, repeat_only=False
#             )
#             emb = self.time_embed(t_emb)
#             print(f"[ControlledUnet] Time embedding (emb) shape: {emb.shape}")
#             h = x.type(self.dtype)
#             print(f"[ControlledUnet] Initial h shape: {h.shape}")
#             for module in self.input_blocks:
#                 h = module(h, emb, context)
#                 hs.append(h)
#             h = self.middle_block(h, emb, context)

#         if control is not None:
#             h += control.pop()

#         if location is not None:
#             loc = location.pop().squeeze(1)
#             loc = repeat(loc, "b d -> b d h w", h=h.shape[-2], w=h.shape[-1])
#             h += loc

#         for i, module in enumerate(self.output_blocks):
#             if only_mid_control or (control is None and location is None):
#                 h = torch.cat([h, hs.pop()], dim=1)
#             else:
#                 if location is not None:
#                     loc = location.pop().squeeze(1)
#                     loc = repeat(loc, "b d -> b d h w", h=h.shape[-2], w=h.shape[-1])
#                     h = torch.cat([h, hs.pop() + control.pop() + loc], dim=1)
#                 else:
                    
#                     h = torch.cat([h, hs.pop() + control.pop()], dim=1)
#             h = module(h, emb, context)

#         h = h.type(x.dtype)
#         return self.out(h)

class ControlledUnetModel(UNetModel): # Inherits from the standard UNetModel
    def forward(
        self,
        x, # Noisy latent
        timesteps=None,
        context=None, # Text embeddings
        control=None, # List of control signals from ControlNet
        location=None, # List of location signals from ControlNet's LocationEncoders
        only_mid_control=False,
        **kwargs,
    ):
        print(f"\n--- ControlledUnetModel Forward ---")
        print(f"[ControlledUnet] Input x shape: {x.shape}")
        if timesteps is not None: print(f"[ControlledUnet] timesteps shape: {timesteps.shape}")
        if context is not None: print(f"[ControlledUnet] context (text_emb) shape: {context.shape}")
        if control is not None: print(f"[ControlledUnet] Received control signals: {len(control)} tensors. Shapes: {[c.shape for c in control]}")
        if location is not None: print(f"[ControlledUnet] Received location signals: {len(location)} tensors. Shapes: {[l.shape for l in location]}")

        hs = []
        with torch.no_grad(): # Original code has this, means encoder part of U-Net isn't trained if sd_locked=True
                              # This seems unusual for the main U-Net unless sd_locked applies elsewhere.
                              # For ControlNet training, UNet is usually frozen.
            t_emb = timestep_embedding(
                timesteps, self.model_channels, repeat_only=False
            )
            emb = self.time_embed(t_emb)
            print(f"[ControlledUnet] Time embedding (emb) shape: {emb.shape}")
            h = x.type(self.dtype)
            print(f"[ControlledUnet] Initial h shape: {h.shape}")
            for i, module in enumerate(self.input_blocks):
                h_prev_shape = h.shape
                h = module(h, emb, context)
                print(f"[ControlledUnet] After input_block {i}: h shape {h.shape} (from {h_prev_shape})")
                hs.append(h)
            h_prev_shape = h.shape
            h = self.middle_block(h, emb, context)
            print(f"[ControlledUnet] After middle_block: h shape {h.shape} (from {h_prev_shape})")

        # Store original middle block output shape for reference
        h_middle_after_unet = h.shape

        if control is not None:
            # ControlNet usually provides controls from deepest to shallowest, so pop() gets the one for middle block
            control_signal_mid = control.pop()
            print(f"[ControlledUnet] Middle control signal shape: {control_signal_mid.shape}, h shape before add: {h.shape}")
            h += control_signal_mid
            print(f"[ControlledUnet] h shape after adding middle control: {h.shape}")


        if location is not None:
            # Location signals also from deepest to shallowest
            loc_signal_mid = location.pop().squeeze(1) # Remove an extra dim if present
            print(f"[ControlledUnet] Middle location signal (raw) shape: {loc_signal_mid.shape}")
            loc_repeated_mid = repeat(loc_signal_mid, "b d -> b d h w", h=h.shape[-2], w=h.shape[-1])
            print(f"[ControlledUnet] Middle location signal (repeated) shape: {loc_repeated_mid.shape}, h shape before add: {h.shape}")
            h += loc_repeated_mid
            print(f"[ControlledUnet] h shape after adding middle location: {h.shape}")


        print(f"[ControlledUnet] Starting output_blocks. h shape: {h.shape}, hs has {len(hs)} skip connections.")
        for i, module in enumerate(self.output_blocks):
            h_prev_shape = h.shape
            skip_conn = hs.pop()
            print(f"[ControlledUnet] Output_block {i}: h_prev {h_prev_shape}, skip_conn {skip_conn.shape}")

            if only_mid_control or (control is None and location is None):
                h = torch.cat([h, skip_conn], dim=1)
                print(f"[ControlledUnet] Output_block {i} (no extra control): h after cat {h.shape}")
            else:
                current_control_signal = control.pop() if control else torch.zeros_like(skip_conn) # Handle if control runs out
                print(f"[ControlledUnet] Output_block {i} current_control_signal shape: {current_control_signal.shape}")

                # Location injection
                if location is not None and len(location) > 0 :
                    loc_signal_out = location.pop().squeeze(1)
                    print(f"[ControlledUnet] Output_block {i} loc_signal_out (raw) shape: {loc_signal_out.shape}")
                    # Skip connection and loc_signal_out might have different spatial dimensions
                    # if loc_signal was for the *input* to this decoder block's corresponding encoder level.
                    # The repeat needs to match the skip_conn's H, W.
                    loc_repeated_out = repeat(loc_signal_out, "b d -> b d h w", h=skip_conn.shape[-2], w=skip_conn.shape[-1])
                    print(f"[ControlledUnet] Output_block {i} loc_signal_out (repeated) shape: {loc_repeated_out.shape}")
                    
                    # Ensure control and loc signals match skip_conn's spatial dims if they are added directly
                    # This original line assumes control.pop() and loc_repeated_out match skip_conn.shape spatially
                    # which might be true if they are the outputs of the ControlNet encoder blocks
                    h_skip_modified = skip_conn + current_control_signal + loc_repeated_out
                    h = torch.cat([h, h_skip_modified], dim=1)
                    print(f"[ControlledUnet] Output_block {i} (with control+loc): h_skip_modified {h_skip_modified.shape}, h after cat {h.shape}")

                else: # No location or location ran out
                    h_skip_modified = skip_conn + current_control_signal
                    h = torch.cat([h, h_skip_modified], dim=1)
                    print(f"[ControlledUnet] Output_block {i} (with control only): h_skip_modified {h_skip_modified.shape}, h after cat {h.shape}")
            
            h_after_cat_shape = h.shape
            h = module(h, emb, context)
            print(f"[ControlledUnet] Output_block {i}: h shape after module {h.shape} (from {h_after_cat_shape})")


        h = h.type(x.dtype)
        output = self.out(h)
        print(f"[ControlledUnet] Final output shape: {output.shape}")
        print(f"--- End ControlledUnetModel Forward ---\n")
        return output


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1)),
        )

        self.loc_blocks = nn.ModuleList([LocationEncoder(out_dim=model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self.loc_blocks.append(LocationEncoder(out_dim=ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch
                self.loc_blocks.append(LocationEncoder(out_dim=ch))

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(  # always uses a self-attn
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self.loc_middle_block = LocationEncoder(out_dim=ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    # def forward(self, x, hint, timesteps, context, location, **kwargs):
    #     t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    #     emb = self.time_embed(t_emb)

    #     guided_hint = self.input_hint_block(hint, emb, context)

    #     outs = []
    #     locs = []

    #     loc_input = location.float().unsqueeze(1)

    #     h = x.type(self.dtype)
    #     for module, zero_conv, loc_module in zip(
    #         self.input_blocks, self.zero_convs, self.loc_blocks
    #     ):
    #         if guided_hint is not None:
    #             h = module(h, emb, context)
    #             h += guided_hint
    #             guided_hint = None
    #         else:
    #             h = module(h, emb, context)
    #         loc_input, loc_zero = loc_module(loc_input, emb.unsqueeze(1))
    #         locs.append(loc_zero)
    #         outs.append(zero_conv(h, emb, context))

    #     h = self.middle_block(h, emb, context)
    #     outs.append(self.middle_block_out(h, emb, context))
    #     locs.append(self.loc_middle_block(loc_input, emb.unsqueeze(1))[1])

    #     return outs, locs

    def forward(self, x, hint, timesteps, context, location, **kwargs):
        print(f"\n--- ControlNet Forward ---")
        print(f"[ControlNet] Input x shape: {x.shape}") # x is noisy latent from U-Net
        print(f"[ControlNet] Input hint shape: {hint.shape}") # Control image (e.g., Canny, OSM)
        if timesteps is not None: print(f"[ControlNet] timesteps shape: {timesteps.shape}")
        if context is not None: print(f"[ControlNet] context (text_emb) shape: {context.shape}")
        if location is not None: print(f"[ControlNet] location (SatCLIP_emb) shape: {location.shape}")

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        print(f"[ControlNet] Time embedding (emb) shape: {emb.shape}")

        guided_hint = self.input_hint_block(hint, emb, context)
        print(f"[ControlNet] guided_hint output shape from input_hint_block: {guided_hint.shape}")

        outs = [] # Control signals for U-Net
        locs = [] # Location signals for U-Net

        loc_input = location.float().unsqueeze(1) # Add sequence dim for LocationEncoder
        print(f"[ControlNet] Initial loc_input (for LocationEncoders) shape: {loc_input.shape}")


        h = x.type(self.dtype) # Noisy latent
        print(f"[ControlNet] Initial h (noisy_latent) shape: {h.shape}")
        for i, (module, zero_conv, loc_module) in enumerate(zip(
            self.input_blocks, self.zero_convs, self.loc_blocks
        )):
            h_prev_shape = h.shape
            if guided_hint is not None:
                # This assumes 'h' and 'guided_hint' are at the same spatial resolution for addition
                # which is true if x (noisy latent) and the output of input_hint_block are compatible
                print(f"[ControlNet] Input_block {i}: Adding guided_hint ({guided_hint.shape}) to h ({h.shape})")
                h_after_module = module(h, emb, context)
                print(f"[ControlNet] Input_block {i}: h_after_module shape {h_after_module.shape}")
                h = h_after_module + guided_hint
                guided_hint = None # Add hint only to the first block's features
            else:
                h = module(h, emb, context)
            
            print(f"[ControlNet] Input_block {i}: h after main module {h.shape} (from {h_prev_shape})")

            loc_input_prev_shape = loc_input.shape
            loc_input, loc_zero = loc_module(loc_input, emb.unsqueeze(1)) # Pass time emb to LocationEncoder
            print(f"[ControlNet] Input_block {i}: loc_input shape after loc_module {loc_input.shape} (from {loc_input_prev_shape}), loc_zero shape {loc_zero.shape}")
            locs.append(loc_zero)

            out_signal = zero_conv(h, emb, context)
            print(f"[ControlNet] Input_block {i}: out_signal (from zero_conv) shape {out_signal.shape}")
            outs.append(out_signal)


        h_prev_shape = h.shape
        h = self.middle_block(h, emb, context)
        print(f"[ControlNet] After middle_block: h shape {h.shape} (from {h_prev_shape})")
        
        out_middle_signal = self.middle_block_out(h, emb, context)
        print(f"[ControlNet] Middle_block_out signal shape: {out_middle_signal.shape}")
        outs.append(out_middle_signal)

        loc_input_prev_shape = loc_input.shape
        loc_middle_zero = self.loc_middle_block(loc_input, emb.unsqueeze(1))[1]
        print(f"[ControlNet] Middle loc_signal shape: {loc_middle_zero.shape} (loc_input was {loc_input_prev_shape})")
        locs.append(loc_middle_zero)

        print(f"[ControlNet] Finished. Returning {len(outs)} control signals and {len(locs)} location signals.")
        print(f"--- End ControlNet Forward ---\n")
        return outs, locs


class ControlLDM(LatentDiffusion):
    def __init__(
        self, control_stage_config, control_key, only_mid_control, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.loc_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        location = batch["location"]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        location = location.to(self.device)
        control = einops.rearrange(control, "b h w c -> b c h w")
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[c], c_concat=[control], c_loc=location)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)

        if cond["c_concat"] is None:
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=None,
                only_mid_control=self.only_mid_control,
            )
        else:
            control, locs = self.control_model(
                x=x_noisy,
                hint=torch.cat(cond["c_concat"], 1),
                location=cond["c_loc"],
                timesteps=t,
                context=cond_txt,
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            locs = [c * scale for c, scale in zip(locs, self.loc_scales)]
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=control,
                location=locs,
                only_mid_control=self.only_mid_control,
            )

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N=4,
        n_row=2,
        sample=False,
        ddim_steps=20,
        ddim_eta=0.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=True,
        plot_denoise_rows=False,
        plot_progressive_rows=True,
        plot_diffusion_rows=False,
        unconditional_guidance_scale=9.0,
        unconditional_guidance_label=None,
        use_ema_scope=True,
        **kwargs,
    ):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c, c_loc = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c["c_loc"][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img(
            (512, 512), batch[self.cond_stage_key], size=16
        )

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, "n b c h w -> b n c h w")
            diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(
                cond={"c_concat": [c_cat], "c_crossattn": [c], "c_loc": c_loc},
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
            )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_loc = c_loc
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross], "c_loc": uc_loc}
            samples_cfg, _ = self.sample_log(
                cond={"c_concat": [c_cat], "c_crossattn": [c], "c_loc": c_loc},
                batch_size=N,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc_full,
            )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(
            ddim_steps, batch_size, shape, cond, verbose=False, **kwargs
        )
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
