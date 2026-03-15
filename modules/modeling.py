from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

import torch
from torch import nn
import torch.nn.functional as F

from modules.until_module import (
    PreTrainedModel,
    AllGather,
    CrossEn,
    GaussianParamHead,
    sample_gaussian,
    mc_match_probability,
    pair_bce_loss,
    gaussian_kl_to_std_normal,
    uniformity_loss,
)
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class LayerNormConv2d(nn.Module):
    def __init__(self, channels):
        super(LayerNormConv2d, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()

class MultiScaleStage(nn.Module):
    def __init__(self, channels, scale):
        super(MultiScaleStage, self).__init__()
        if scale == 1.0:
            self.resample = nn.Identity()
        elif scale < 1.0:
            pool_kernel = int(round(1.0 / scale))
            self.resample = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel)
        else:
            self.resample = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)

        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            LayerNormConv2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            LayerNormConv2d(channels),
        )

    def forward(self, x):
        x = self.resample(x)
        return self.proj(x)

class TemporalResidualBlock(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(TemporalResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)
        if self.loose_type and self.sim_header != "tightTransf":
            self.use_pcme_prob = True
            self.muse_num_frames = 12
            self.muse_tokens_per_frame = 50
            self.muse_scales = [0.5, 1.0, 2.0]
            self.muse_stages = nn.ModuleList(
                [MultiScaleStage(transformer_width, scale=scale) for scale in self.muse_scales]
            )
            self.muse_temporal_blocks = nn.ModuleList(
                [TemporalResidualBlock(transformer_width, dropout=0.1) for _ in range(4)]
            )
            logsigma_min = getattr(self.task_config, "pcme_logsigma_min", -7.0)
            logsigma_max = getattr(self.task_config, "pcme_logsigma_max", 7.0)
            logsigma_bias_init = getattr(self.task_config, "pcme_logsigma_bias_init", -5.0)
            self.txt_prob_head = GaussianParamHead(
                transformer_width,
                logsigma_min=logsigma_min,
                logsigma_max=logsigma_max,
                logsigma_bias_init=logsigma_bias_init,
            )
            self.vid_prob_head = GaussianParamHead(
                transformer_width,
                logsigma_min=logsigma_min,
                logsigma_max=logsigma_max,
                logsigma_bias_init=logsigma_bias_init,
            )
            self.pcme_alpha = nn.Parameter(torch.tensor(float(getattr(self.task_config, "pcme_alpha_init", 1.0))))
            self.pcme_beta = nn.Parameter(torch.tensor(float(getattr(self.task_config, "pcme_beta_init", 0.0))))
            show_log(task_config, "\t pcme_train_samples: {}".format(getattr(self.task_config, "pcme_train_samples", 4)))
            show_log(task_config, "\t pcme_eval_samples: {}".format(getattr(self.task_config, "pcme_eval_samples", 16)))
        else:
            self.use_pcme_prob = False

        self.current_epoch = 0
        self.latest_debug_stats = {}
        self.loss_fct = CrossEn()

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True, video_frame=video_frame)

        if self.training:
            sim_matrix, _, aux_info = self.get_similarity_logits(
                sequence_output,
                visual_output,
                attention_mask,
                video_mask,
                shaped=True,
                loose_type=self.loose_type,
                return_aux=True,
            )
            loss_ce = (self.loss_fct(sim_matrix) + self.loss_fct(sim_matrix.T)) / 2
            loss = loss_ce

            if self.use_pcme_prob and aux_info is not None and self._should_enable_aux_loss():
                prob_matrix = aux_info["match_prob"]
                if prob_matrix is None or aux_info["txt_mu"] is None or aux_info["vid_mu"] is None:
                    prob_matrix = None
                if prob_matrix is not None:
                    if prob_matrix.size(0) != prob_matrix.size(1):
                        raise ValueError("PCME mixed loss expects a square similarity matrix during training.")
                    target_matrix = torch.eye(prob_matrix.size(0), device=prob_matrix.device, dtype=prob_matrix.dtype)
                    loss_bce = pair_bce_loss(prob_matrix, target_matrix)
                    loss_kl = gaussian_kl_to_std_normal(aux_info["txt_mu"], aux_info["txt_log_sigma"])
                    loss_kl = loss_kl + gaussian_kl_to_std_normal(aux_info["vid_mu"], aux_info["vid_log_sigma"])
                    temperature = getattr(self.task_config, "pcme_uniformity_t", 2.0)
                    loss_unif = uniformity_loss(aux_info["txt_mu"], temperature)
                    loss_unif = loss_unif + uniformity_loss(aux_info["vid_mu"], temperature)

                    lambda_scale = self._get_aux_lambda_scale()
                    lambda_match = getattr(self.task_config, "pcme_lambda_match", 1.0)
                    lambda_kl = getattr(self.task_config, "pcme_lambda_kl", 5e-4)
                    lambda_unif = getattr(self.task_config, "pcme_lambda_unif", 1e-3)
                    loss = loss + lambda_scale * (
                        lambda_match * loss_bce + lambda_kl * loss_kl + lambda_unif * loss_unif
                    )

            with torch.no_grad():
                diag_vals = torch.diagonal(sim_matrix)
                sim_sum = sim_matrix.sum()
                offdiag_den = max(sim_matrix.numel() - diag_vals.numel(), 1)
                offdiag_mean = (sim_sum - diag_vals.sum()) / offdiag_den
                logsigma_mean = None
                if aux_info is not None and aux_info.get("txt_log_sigma") is not None and aux_info.get("vid_log_sigma") is not None:
                    logsigma_mean = 0.5 * (
                        aux_info["txt_log_sigma"].mean().item() + aux_info["vid_log_sigma"].mean().item()
                    )
                self.latest_debug_stats = {
                    "muse_mix": float(aux_info.get("muse_mix", 0.0)) if aux_info is not None else 0.0,
                    "prob_mix": float(aux_info.get("prob_mix", 0.0)) if aux_info is not None else 0.0,
                    "diag_mean": float(diag_vals.mean().item()),
                    "offdiag_mean": float(offdiag_mean.item()),
                    "diag_minus_offdiag": float((diag_vals.mean() - offdiag_mean).item()),
                    "logsigma_mean": logsigma_mean,
                }

            return loss
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids).float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        if hasattr(self, "muse_stages"):
            _, visual_hidden_tokens = self.clip.encode_image(video, return_hidden=True, video_frame=video_frame)
            visual_hidden_tokens = visual_hidden_tokens.float()
            visual_hidden = visual_hidden_tokens.view(
                bs_pair, video_frame * visual_hidden_tokens.size(1), visual_hidden_tokens.size(-1)
            )
        else:
            visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
            visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return sequence_output, visual_output

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _linear_ramp(self, target, warmup_epochs, ramp_epochs, epoch_idx):
        if epoch_idx < warmup_epochs:
            return 0.0
        if ramp_epochs <= 0:
            return float(target)
        ramp_pos = min(epoch_idx - warmup_epochs + 1, ramp_epochs)
        return float(target) * float(ramp_pos) / float(ramp_epochs)

    def _get_muse_mix(self):
        target = getattr(self.task_config, "muse_mix_target", 0.5)
        warmup_epochs = getattr(self.task_config, "muse_warmup_epochs", 2)
        ramp_epochs = getattr(self.task_config, "muse_ramp_epochs", 3)
        return self._linear_ramp(target, warmup_epochs, ramp_epochs, self.current_epoch)

    def _get_prob_mix(self):
        mode = getattr(self.task_config, "pcme_mode", "deterministic")
        if mode == "deterministic":
            return 0.0
        if mode == "probabilistic":
            return 1.0

        target = getattr(self.task_config, "pcme_prob_mix_target", 0.3)
        warmup_epochs = getattr(self.task_config, "pcme_prob_warmup_epochs", 4)
        ramp_epochs = getattr(self.task_config, "pcme_prob_ramp_epochs", 2)
        return self._linear_ramp(target, warmup_epochs, ramp_epochs, self.current_epoch)

    def _get_aux_lambda_scale(self):
        warmup_epochs = getattr(self.task_config, "pcme_aux_warmup_epochs", 4)
        if self.current_epoch < warmup_epochs:
            return 0.0
        if self.current_epoch == warmup_epochs:
            return 0.5
        return 1.0

    def _should_enable_aux_loss(self):
        if not getattr(self.task_config, "pcme_enable_aux_loss", True):
            return False
        if getattr(self.task_config, "pcme_mode", "deterministic") == "deterministic":
            return False
        return self._get_aux_lambda_scale() > 0.0

    def _muse_lite_fuse_visual_tokens(self, visual_output, video_mask):
        batch_size, total_tokens, channels = visual_output.shape
        num_frames = video_mask.size(1)

        if num_frames != self.muse_num_frames:
            raise ValueError("MUSE expects {} frames, but got {}.".format(self.muse_num_frames, num_frames))
        if total_tokens % num_frames != 0:
            raise ValueError("Invalid visual token layout: total_tokens={} is not divisible by frames={}.".format(
                total_tokens, num_frames
            ))

        tokens_per_frame = total_tokens // num_frames
        if tokens_per_frame != self.muse_tokens_per_frame:
            raise ValueError("MUSE expects {} tokens per frame, but got {}.".format(
                self.muse_tokens_per_frame, tokens_per_frame
            ))

        visual_output = visual_output.view(batch_size, num_frames, tokens_per_frame, channels)
        cls_tokens = visual_output[:, :, 0, :].contiguous()
        patch_tokens = visual_output[:, :, 1:, :].contiguous()

        spatial_tokens = patch_tokens.size(2)
        spatial_size = int(math.sqrt(spatial_tokens))
        if spatial_size * spatial_size != spatial_tokens:
            raise ValueError("Patch token count {} is not a square number.".format(spatial_tokens))

        patch_map = patch_tokens.view(batch_size * num_frames, spatial_size, spatial_size, channels).permute(0, 3, 1, 2)

        multiscale_tokens = []
        for stage in self.muse_stages:
            stage_feature = stage(patch_map)
            stage_feature = stage_feature.view(batch_size, num_frames, channels, -1).permute(0, 1, 3, 2).contiguous()
            multiscale_tokens.append(stage_feature.view(batch_size, -1, channels))

        fused_tokens = torch.cat([cls_tokens, torch.cat(multiscale_tokens, dim=1)], dim=1)
        for block in self.muse_temporal_blocks:
            fused_tokens = block(fused_tokens)

        return fused_tokens[:, :num_frames, :].contiguous()

    def _compute_loose_global_embeddings(self, sequence_output, visual_output, video_mask, sim_header="meanP"):
        if sim_header in ["meanP", "seqLSTM", "seqTransf"]:
            sim_header = "MUSE"

        if sim_header != "MUSE":
            raise ValueError("Unsupported loose similarity header: {}".format(sim_header))

        batch_size, total_tokens, channels = visual_output.shape
        num_frames = video_mask.size(1)
        if total_tokens % num_frames != 0:
            raise ValueError(
                "Invalid visual token layout: total_tokens={} is not divisible by frames={}.".format(
                    total_tokens, num_frames
                )
            )
        tokens_per_frame = total_tokens // num_frames
        visual_tokens = visual_output.view(batch_size, num_frames, tokens_per_frame, channels)
        frame_cls_tokens = visual_tokens[:, :, 0, :].contiguous()
        frame_cls_tokens = frame_cls_tokens / frame_cls_tokens.norm(dim=-1, keepdim=True)
        video_embed_base = self._mean_pooling_for_similarity_visual(frame_cls_tokens, video_mask)
        video_embed_base = video_embed_base / video_embed_base.norm(dim=-1, keepdim=True)

        muse_mix = self._get_muse_mix()
        if muse_mix <= 0.0:
            video_embed = video_embed_base
        else:
            visual_output_muse = self._muse_lite_fuse_visual_tokens(visual_output, video_mask)
            visual_output_muse = visual_output_muse / visual_output_muse.norm(dim=-1, keepdim=True)
            video_embed_muse = self._mean_pooling_for_similarity_visual(visual_output_muse, video_mask)
            video_embed_muse = video_embed_muse / video_embed_muse.norm(dim=-1, keepdim=True)
            video_embed = (1.0 - muse_mix) * video_embed_base + muse_mix * video_embed_muse
            video_embed = video_embed / video_embed.norm(dim=-1, keepdim=True)

        text_embed = sequence_output.squeeze(1)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        return text_embed, video_embed, muse_mix

    def _probabilistic_loose_similarity(self, text_embed, video_embed, prob_mix, return_aux=False):
        if self.training:
            text_embed = allgather(text_embed, self.task_config)
            video_embed = allgather(video_embed, self.task_config)
            torch.distributed.barrier()

        mode = getattr(self.task_config, "pcme_mode", "deterministic")
        det_logits = self.clip.logit_scale.exp() * torch.matmul(text_embed, video_embed.t())
        sim_logits = det_logits
        txt_mu, txt_log_sigma = None, None
        vid_mu, vid_log_sigma = None, None
        match_prob = None
        prob_logits = None

        requires_prob = mode == "probabilistic" or prob_mix > 0.0 or self._should_enable_aux_loss()
        if requires_prob:
            txt_mu, txt_log_sigma = self.txt_prob_head(text_embed)
            vid_mu, vid_log_sigma = self.vid_prob_head(video_embed)

            num_samples = getattr(self.task_config, "pcme_train_samples", 4) if self.training \
                else getattr(self.task_config, "pcme_eval_samples", 16)
            txt_samples = sample_gaussian(txt_mu, txt_log_sigma, num_samples)
            vid_samples = sample_gaussian(vid_mu, vid_log_sigma, num_samples)

            alpha = F.softplus(self.pcme_alpha) + 1e-6
            beta = self.pcme_beta
            match_prob, prob_logits = mc_match_probability(txt_samples, vid_samples, alpha=alpha, beta=beta)

            if mode == "probabilistic":
                sim_logits = prob_logits
            elif mode == "hybrid":
                sim_logits = (1.0 - prob_mix) * det_logits + prob_mix * prob_logits

        aux_info = None
        if return_aux:
            aux_info = {
                "match_prob": match_prob,
                "txt_mu": txt_mu,
                "txt_log_sigma": txt_log_sigma,
                "vid_mu": vid_mu,
                "vid_log_sigma": vid_log_sigma,
            }
        return sim_logits, aux_info

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP", return_aux=False):
        del attention_mask
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        text_embed, video_embed, muse_mix = self._compute_loose_global_embeddings(
            sequence_output, visual_output, video_mask, sim_header=sim_header
        )
        prob_mix = self._get_prob_mix()
        retrieve_logits, aux_info = self._probabilistic_loose_similarity(
            text_embed, video_embed, prob_mix=prob_mix, return_aux=return_aux
        )
        if return_aux and aux_info is not None:
            aux_info["muse_mix"] = muse_mix
            aux_info["prob_mix"] = prob_mix
        return retrieve_logits, aux_info

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, loose_type=False, return_aux=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        aux_info = None
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf", "MUSE"]
            retrieve_logits, aux_info = self._loose_similarity(
                sequence_output,
                visual_output,
                attention_mask,
                video_mask,
                sim_header=self.sim_header,
                return_aux=return_aux,
            )
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

        return retrieve_logits, contrastive_direction, aux_info
