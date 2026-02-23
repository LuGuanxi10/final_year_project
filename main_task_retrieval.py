from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT

torch.distributed.init_process_group(backend="nccl")

global logger

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local-rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf", "MUSE"],
                        help="choice a similarity header.")
    parser.add_argument('--pcme_train_samples', type=int, default=4, help='MC samples for training.')
    parser.add_argument('--pcme_eval_samples', type=int, default=16, help='MC samples for evaluation.')
    parser.add_argument('--pcme_logsigma_min', type=float, default=-7.0, help='Lower clamp bound of log sigma.')
    parser.add_argument('--pcme_logsigma_max', type=float, default=7.0, help='Upper clamp bound of log sigma.')
    parser.add_argument('--pcme_alpha_init', type=float, default=1.0, help='Initial alpha for probabilistic matching.')
    parser.add_argument('--pcme_beta_init', type=float, default=0.0, help='Initial beta for probabilistic matching.')
    parser.add_argument('--pcme_lambda_match', type=float, default=1.0, help='Weight for match BCE loss.')
    parser.add_argument('--pcme_lambda_kl', type=float, default=5e-4, help='Weight for Gaussian KL loss.')
    parser.add_argument('--pcme_lambda_unif', type=float, default=1e-3, help='Weight for uniformity loss.')
    parser.add_argument('--pcme_uniformity_t', type=float, default=2.0, help='Temperature for uniformity regularizer.')
    parser.add_argument('--pcme_mode', type=str, default="deterministic",
                        choices=["deterministic", "hybrid", "probabilistic"],
                        help='Similarity mode for loose retrieval.')
    parser.add_argument('--muse_mix_target', type=float, default=0.5, help='Target mix ratio for MUSE fused video embedding.')
    parser.add_argument('--muse_warmup_epochs', type=int, default=2, help='Warmup epochs before enabling MUSE mix.')
    parser.add_argument('--muse_ramp_epochs', type=int, default=3, help='Epochs to ramp MUSE mix from 0 to target.')
    parser.add_argument('--pcme_prob_mix_target', type=float, default=0.3, help='Target mix ratio of probabilistic logits.')
    parser.add_argument('--pcme_prob_warmup_epochs', type=int, default=4, help='Warmup epochs before probabilistic mix.')
    parser.add_argument('--pcme_prob_ramp_epochs', type=int, default=2, help='Epochs to ramp probabilistic mix.')
    parser.add_argument('--pcme_enable_aux_loss', type=str2bool, default=True,
                        help='Whether to enable PCME auxiliary BCE/KL/Uniformity losses.')
    parser.add_argument('--pcme_aux_warmup_epochs', type=int, default=4, help='Warmup epochs before auxiliary losses.')
    parser.add_argument('--pcme_logsigma_bias_init', type=float, default=-5.0, help='Initial bias for log sigma projection.')
    parser.add_argument('--lr_clip', type=float, default=5e-6, help='Learning rate for clip.* parameters.')
    parser.add_argument('--lr_new_modules', type=float, default=3e-4, help='Learning rate for MUSE/PCME modules.')

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    args = parser.parse_args()

    legacy_loose_headers = {"meanP", "seqLSTM", "seqTransf"}
    if args.sim_header == "tightTransf":
        args.loose_type = False
    else:
        if args.sim_header == "MUSE" and not args.loose_type:
            print("[compat] Enabling --loose_type because MUSE is a loose retrieval header.")
            args.loose_type = True

        if args.loose_type and args.sim_header in legacy_loose_headers:
            print("[compat] --sim_header {} is mapped to MUSE in loose retrieval mode.".format(args.sim_header))
            args.sim_header = "MUSE"

    if args.sim_header == "MUSE" and args.max_frames != 12:
        raise ValueError("MUSE requires --max_frames=12, but got {}.".format(args.max_frames))

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if not 0.0 <= args.muse_mix_target <= 1.0:
        raise ValueError("--muse_mix_target must be in [0, 1], got {}".format(args.muse_mix_target))
    if not 0.0 <= args.pcme_prob_mix_target <= 1.0:
        raise ValueError("--pcme_prob_mix_target must be in [0, 1], got {}".format(args.pcme_prob_mix_target))
    if args.muse_warmup_epochs < 0 or args.muse_ramp_epochs < 0:
        raise ValueError("--muse_warmup_epochs and --muse_ramp_epochs must be >= 0")
    if args.pcme_prob_warmup_epochs < 0 or args.pcme_prob_ramp_epochs < 0:
        raise ValueError("--pcme_prob_warmup_epochs and --pcme_prob_ramp_epochs must be >= 0")
    if args.pcme_aux_warmup_epochs < 0:
        raise ValueError("--pcme_aux_warmup_epochs must be >= 0")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

"""
def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu
"""

def init_device(args, local_rank):
    global logger

    # ensure local_rank is int
    local_rank = int(local_rank)

    if torch.cuda.is_available():
        # 把当前进程绑定到对应的本地 GPU
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpu = 0

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    # 如果有 GPU，检查 batch_size 是否和 GPU 数匹配（原逻辑）
    if n_gpu > 0:
        if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
            raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu


def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_module_markers = ("muse_", "txt_prob_head", "vid_prob_head", "pcme_alpha", "pcme_beta")

    def _group_lr(param_name):
        if param_name.startswith("clip."):
            return args.lr_clip
        if any(marker in param_name for marker in new_module_markers):
            return args.lr_new_modules
        return args.lr

    grouped_params = {}
    for n, p in param_optimizer:
        if not p.requires_grad:
            continue
        decay_key = "no_decay" if any(nd in n for nd in no_decay) else "decay"
        lr = _group_lr(n)
        key = (lr, decay_key)
        grouped_params.setdefault(key, []).append(p)

    weight_decay = 0.2
    optimizer_grouped_parameters = []
    for (lr, decay_key), params in sorted(grouped_params.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len(params) == 0:
            continue
        optimizer_grouped_parameters.append({
            'params': params,
            'weight_decay': 0.0 if decay_key == "no_decay" else weight_decay,
            'lr': lr
        })

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
    #                                                  output_device=local_rank, find_unused_parameters=True)
    # 确保 model 在正确的 device 上（init_model 通常已做，但这里再保证）
    if device.type == 'cuda':
        torch.cuda.set_device(int(local_rank))
    model.to(device)

    # 最后 wrap DDP（确保 device_ids/output_device 使用 local_rank）
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[int(local_rank)],
        output_device=int(local_rank),
        find_unused_parameters=True
    )


    return optimizer, scheduler, model

def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask = batch
        loss = model(input_ids, segment_ids, input_mask, video, video_mask)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                lr_values = sorted(list(set(float(group['lr']) for group in optimizer.param_groups)))
                lr_text = "-".join(['%.9f' % itm for itm in lr_values])
                model_ref = model.module if hasattr(model, 'module') else model
                debug_stats = getattr(model_ref, "latest_debug_stats", {}) or {}
                logit_scale = float(model_ref.clip.logit_scale.exp().item()) if hasattr(model_ref, "clip") else 0.0
                if debug_stats:
                    logsigma_mean = debug_stats.get("logsigma_mean", None)
                    logsigma_text = "NA" if logsigma_mean is None else "{:.4f}".format(logsigma_mean)
                    logger.info(
                        "Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f, "
                        "muse_mix: %.3f, prob_mix: %.3f, diag-offdiag: %.4f, logsigma_mean: %s, logit_scale: %.4f",
                        epoch + 1, args.epochs, step + 1, len(train_dataloader), lr_text, float(loss),
                        (time.time() - start_time) / (log_step * args.gradient_accumulation_steps),
                        float(debug_stats.get("muse_mix", 0.0)),
                        float(debug_stats.get("prob_mix", 0.0)),
                        float(debug_stats.get("diag_minus_offdiag", 0.0)),
                        logsigma_text,
                        logit_scale,
                    )
                else:
                    logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f, logit_scale: %.4f",
                                epoch + 1, args.epochs, step + 1, len(train_dataloader), lr_text, float(loss),
                                (time.time() - start_time) / (log_step * args.gradient_accumulation_steps),
                                logit_scale)
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def set_model_epoch(model, epoch):
    model_ref = model.module if hasattr(model, 'module') else model
    if hasattr(model_ref, "current_epoch"):
        model_ref.current_epoch = int(epoch)

def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask,
                                                                     loose_type=model.loose_type)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix

def eval_epoch(args, model, test_dataloader, device, n_gpu):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch

            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        if n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_t_output_splits = []
            batch_v_output_splits = []
            bacth_len = len(batch_list_t)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                    batch_list_v_splits.append(devc_batch_list)

                    devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                      batch_t_output_splits[dev_id], batch_v_output_splits[dev_id]) for dev_id in device_ids]
            parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
            sim_matrix = []
            for idx in range(len(parallel_outputs)):
                sim_matrix += parallel_outputs[idx]
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        else:
            sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']
    return R1

def main():
    global logger
    args = get_args()

    # ------------------ torchrun / LOCAL_RANK 兼容补丁 ------------------
    # 确保 args.local_rank 来自 torchrun 设置的环境变量 LOCAL_RANK（优先使用 env）
    import os
    # 如果 torchrun 启动，会在环境变量中设置 LOCAL_RANK；优先取该值
    args.local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))

    # 打印进程使用的本地 rank（有助于调试）
    print(f"[main] LOCAL_RANK env -> args.local_rank = {args.local_rank}")
    # -------------------------------------------------------------------

    args = set_seed_logger(args)
    if args.local_rank == 0:
        logger.info("Effective retrieval header: %s (loose_type=%s, max_frames=%d)",
                    args.sim_header, args.loose_type, args.max_frames)
        logger.info("PCME config: mode=%s train_samples=%d eval_samples=%d alpha_init=%.4f beta_init=%.4f "
                    "lambda_match=%.6f lambda_kl=%.6f lambda_unif=%.6f uniformity_t=%.3f enable_aux=%s aux_warmup=%d",
                    args.pcme_mode,
                    args.pcme_train_samples, args.pcme_eval_samples, args.pcme_alpha_init, args.pcme_beta_init,
                    args.pcme_lambda_match, args.pcme_lambda_kl, args.pcme_lambda_unif, args.pcme_uniformity_t,
                    str(args.pcme_enable_aux_loss), args.pcme_aux_warmup_epochs)
        logger.info("Mix schedule: muse_target=%.3f warmup=%d ramp=%d | prob_target=%.3f warmup=%d ramp=%d",
                    args.muse_mix_target, args.muse_warmup_epochs, args.muse_ramp_epochs,
                    args.pcme_prob_mix_target, args.pcme_prob_warmup_epochs, args.pcme_prob_ramp_epochs)
        logger.info("LR schedule: base=%.8f clip=%.8f new_modules=%.8f", args.lr, args.lr_clip, args.lr_new_modules)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert  args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model is not None and os.path.exists(args.resume_model):
            print(f"=> Loading checkpoint from {args.resume_model}")

            checkpoint = torch.load(args.resume_model, map_location='cpu')

            # ---- Restore Model ----
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("=> Model weights loaded (wrapped state_dict).")
            else:
                try:
                    model.load_state_dict(checkpoint, strict=False)
                    print("=> Model weights loaded (raw state_dict).")
                except:
                    print("❌ ERROR: Can't load model weights from checkpoint")
                    raise

            # ---- Restore Optimizer ----
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("=> Optimizer state restored.")
                except Exception as e:
                    print(f"=> WARNING: Failed to restore optimizer. Training optimizer fresh. Reason: {e}")
            else:
                print("=> No optimizer state in checkpoint. Starting new optimizer.")

            # ---- Restore Epoch ----
            if 'epoch' in checkpoint:
                resumed_epoch = int(checkpoint['epoch']) + 1
                print(f"=> Resuming from epoch {resumed_epoch}")
            else:
                resumed_epoch = 1
                print("=> No epoch stored in checkpoint. Starting from epoch 1")

            # ---- Restore Scheduler if exists ----
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("=> Scheduler state restored.")
                except:
                    print("=> WARNING: Scheduler state failed to load, using fresh scheduler.")
        else:
            resumed_epoch = 0
            print("=> No resume checkpoint provided. Training from scratch.")
            # resumed_loss = checkpoint['loss']
        
        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            set_model_epoch(model, epoch)
            train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            prob_mix_target_epoch = args.pcme_prob_mix_target
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

                ## Run on val dataset, this process is *TIME-consuming*.
                # logger.info("Eval on val dataset")
                # R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)

                R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
                if (epoch + 1) == 4 and R1 < 35.0 and args.pcme_prob_mix_target > 0.2:
                    prob_mix_target_epoch = 0.2
                    logger.warning(
                        "Epoch 4 R1=%.4f is below 35.0, reduce pcme_prob_mix_target to %.3f for remaining epochs.",
                        R1, prob_mix_target_epoch
                    )
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

            prob_mix_target_tensor = torch.tensor([prob_mix_target_epoch], dtype=torch.float32, device=device)
            torch.distributed.broadcast(prob_mix_target_tensor, src=0)
            synced_prob_mix_target = float(prob_mix_target_tensor.item())
            if synced_prob_mix_target != args.pcme_prob_mix_target:
                args.pcme_prob_mix_target = synced_prob_mix_target
                model_ref = model.module if hasattr(model, 'module') else model
                if hasattr(model_ref, "task_config"):
                    model_ref.task_config.pcme_prob_mix_target = synced_prob_mix_target

        ## Uncomment if want to test on the best checkpoint
        # if args.local_rank == 0:
        #     model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
        #     eval_epoch(args, model, test_dataloader, device, n_gpu)

    elif args.do_eval:
        set_model_epoch(model, max(args.epochs - 1, 0))
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu)

if __name__ == "__main__":
    main()
