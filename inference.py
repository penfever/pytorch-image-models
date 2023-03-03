#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import json
import logging
import os
import time
from contextlib import suppress
from functools import partial

import numpy as np
import pandas as pd
import torch

from timm.data import create_dataset, create_loader, resolve_data_config, ImageNetInfo, infer_imagenet_subset
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging, set_jit_fuser, ParseKwargs

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

IN100_TRUE_IDX = [386, 928, 931, 704, 907, 291, 454, 76, 952, 788, 245, 937, 924, 8, 983, 816, 920, 379, 204, 396, 929, 619, 815, 88, 84, 217, 118, 935, 987, 642, 950, 951, 954, 557, 18, 967, 945, 6, 440, 348, 22, 571, 23, 963, 104, 958, 579, 312, 534, 620, 115, 298, 284, 552, 373, 997, 182, 422, 308, 839, 13, 489, 805, 832, 85, 695, 2, 863, 310, 565, 886, 455, 988, 347, 580, 425, 99, 424, 105, 107, 343, 658, 721, 443, 421, 679, 19, 825, 130, 309, 849, 879, 496, 971, 922, 985, 286, 625, 637, 943]
ir_idx = [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 
125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 
231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 
296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 
362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 
463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 
637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 
889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 
983, 988]
IN100_DOGS_IDX = [151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250]
ia_idx = [6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107, 108, 110, 
113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307, 308, 309, 
310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386, 397, 400, 401, 
402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483, 486, 488, 492, 496, 
514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614, 626, 627, 
640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773, 774, 776, 
779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870, 879, 880, 
888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980, 981, 984, 
986, 987, 988]
common_ia_idx = [n for n in ia_idx if n in IN100_TRUE_IDX]
common_ir_idx = [n for n in ir_idx if n in IN100_TRUE_IDX]
common_dir_idx = [n for n in ir_idx if n in IN100_DOGS_IDX]
common_dia_idx = [n for n in ia_idx if n in IN100_DOGS_IDX]

_FMT_EXT = {
    'json': '.json',
    'json-record': '.json',
    'json-split': '.json',
    'parquet': '.parquet',
    'csv': '.csv',
}

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--in100', action='store_true', default=False,
                    help='select in100 subset of ImageNet (default: False)')
parser.add_argument('--in100d', action='store_true', default=False,
                    help='select in100d subset of ImageNet (default: False)')
parser.add_argument('--in100dr', action='store_true', default=False,
                    help='select in100dr subset of ImageNet (default: False)')
parser.add_argument('--in100r', action='store_true', default=False,
                    help='select in100r subset of ImageNet (default: False)')
parser.add_argument('--in100da', action='store_true', default=False,
                    help='select in100da subset of ImageNet (default: False)')
parser.add_argument('--in100a', action='store_true', default=False,
                    help='select in100a subset of ImageNet (default: False)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=False,
                    help='use Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)

scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")

parser.add_argument('--results-dir', type=str, default=None,
                    help='folder for output results')
parser.add_argument('--results-file', type=str, default=None,
                    help='results filename (relative to results-dir)')
parser.add_argument('--results-format', type=str, nargs='+', default=['csv'],
                    help='results format (one of "csv", "json", "json-split", "parquet")')
parser.add_argument('--results-separate-col', action='store_true', default=False,
                    help='separate output columns per result index.')
parser.add_argument('--topk', default=1, type=int,
                    metavar='N', help='Top-k to output to CSV')
parser.add_argument('--fullname', action='store_true', default=False,
                    help='use full sample name in output (not just basename).')
parser.add_argument('--filename-col', type=str, default='filename',
                    help='name for filename / sample name column')
parser.add_argument('--index-col', type=str, default='index',
                    help='name for output indices column(s)')
parser.add_argument('--label-col', type=str, default='label',
                    help='name for output indices column(s)')
parser.add_argument('--output-col', type=str, default=None,
                    help='name for logit/probs output column(s)')
parser.add_argument('--output-type', type=str, default='prob',
                    help='output type colum ("prob" for probabilities, "logit" for raw logits)')
parser.add_argument('--label-type', type=str, default='description',
                    help='type of label to output, one of  "none", "name", "description", "detailed"')
parser.add_argument('--include-index', action='store_true', default=False,
                    help='include the class index in results')
parser.add_argument('--exclude-output', action='store_true', default=False,
                    help='exclude logits/probs from results, just indices. topk must be set !=0.')


def main():
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # resolve AMP arguments based on PyTorch / Apex availability
    amp_autocast = suppress
    if args.amp:
        assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
        assert args.amp_dtype in ('float16', 'bfloat16')
        amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        _logger.info('Running inference in mixed precision with native PyTorch AMP.')
    else:
        _logger.info('Running inference in float32. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=in_chans,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        **args.model_kwargs,
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    _logger.info(
        f'Model {args.model} created, param count: {sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device)
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        model = torch.jit.script(model)
    elif args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    root_dir = args.data or args.data_dir
    dataset = create_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        class_map=args.class_map,
    )

    if test_time_pool:
        data_config['crop_pct'] = 1.0

    workers = 1 if 'tfds' in args.dataset or 'wds' in args.dataset else args.workers
    loader = create_loader(
        dataset,
        batch_size=args.batch_size,
        use_prefetcher=True,
        num_workers=workers,
        **data_config,
    )

    to_label = None
    if args.label_type in ('name', 'description', 'detail'):
        imagenet_subset = infer_imagenet_subset(model)
        if imagenet_subset is not None:
            dataset_info = ImageNetInfo(imagenet_subset)
            if args.label_type == 'name':
                to_label = lambda x: dataset_info.index_to_label_name(x)
            elif args.label_type == 'detail':
                to_label = lambda x: dataset_info.index_to_description(x, detailed=True)
            else:
                to_label = lambda x: dataset_info.index_to_description(x)
            to_label = np.vectorize(to_label)
        else:
            _logger.error("Cannot deduce ImageNet subset from model, no labelling will be performed.")

    top_k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    all_indices = []
    all_labels = []
    all_outputs = []
    all_targets = []
    use_probs = args.output_type == 'prob'
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if args.in100r or args.in100dr:
                ir_idx_np = np.array(ir_idx)
                target = ir_idx_np[target.cpu().numpy()]
                all_targets.append(target)
            elif args.in100a or args.in100da:
                ia_idx_np = np.array(ia_idx)
                target = ia_idx_np[target.cpu().numpy()]
                all_targets.append(target)
            else:
                all_targets.append(target.cpu().numpy())
            with amp_autocast():
                output = model(input)
            if args.in100:
                not_icap_idx = [i for i in range(1000) if i not in IN100_TRUE_IDX]
                output[:, not_icap_idx] = float("-inf")
            elif args.in100r:
                not_icap_idx = [i for i in range(1000) if i not in common_ir_idx]
                output[:, not_icap_idx] = float("-inf")
            elif args.in100a:
                not_icap_idx = [i for i in range(1000) if i not in common_ia_idx]
                output[:, not_icap_idx] = float("-inf")
            elif args.in100d:
                not_icap_idx = [i for i in range(1000) if i not in IN100_DOGS_IDX]
                output[:, not_icap_idx] = float("-inf")
            elif args.in100dr:
                not_icap_idx = [i for i in range(1000) if i not in common_dir_idx]
                output[:, not_icap_idx] = float("-inf")
            elif args.in100da:
                not_icap_idx = [i for i in range(1000) if i not in common_dia_idx]
                output[:, not_icap_idx] = float("-inf")
            if use_probs:
                output = output.softmax(-1)
            if top_k:
                output, indices = output.topk(top_k)
                np_indices = indices.cpu().numpy()
                if args.include_index:
                    all_indices.append(np_indices)
                if to_label is not None:
                    np_labels = to_label(np_indices)
                    all_labels.append(np_labels)

            all_outputs.append(output.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

    all_indices = np.concatenate(all_indices, axis=0) if all_indices else None
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    all_outputs = np.concatenate(all_outputs, axis=0).astype(np.float32)
    all_targets = np.concatenate(all_targets, axis=0).astype(np.int64)
    filenames = loader.dataset.filenames(basename=not args.fullname)

    output_col = args.output_col or ('prob' if use_probs else 'logit')
    data_dict = {args.filename_col: filenames}
    if args.results_separate_col and all_outputs.shape[-1] > 1:
        if all_indices is not None:
            for i in range(all_indices.shape[-1]):
                data_dict[f'{args.index_col}_{i}'] = all_indices[:, i]
        if all_labels is not None:
            for i in range(all_labels.shape[-1]):
                data_dict[f'{args.label_col}_{i}'] = all_labels[:, i]
        for i in range(all_outputs.shape[-1]):
            data_dict[f'{output_col}_{i}'] = all_outputs[:, i]
        for i in range(all_targets.shape[-1]):
            data_dict[f'target_{i}'] = all_targets[:, i]
    else:
        if all_indices is not None:
            if all_indices.shape[-1] == 1:
                all_indices = all_indices.squeeze(-1)
            data_dict[args.index_col] = list(all_indices)
        if all_labels is not None:
            if all_labels.shape[-1] == 1:
                all_labels = all_labels.squeeze(-1)
            data_dict[args.label_col] = list(all_labels)
        if all_outputs.shape[-1] == 1:
            all_outputs = all_outputs.squeeze(-1)
        data_dict[output_col] = list(all_outputs)
        if all_targets.shape[-1] == 1:
            all_targets = all_targets.squeeze(-1)
        data_dict["targets"] = list(all_targets)
    df = pd.DataFrame(data=data_dict)

    results_filename = args.results_file
    if results_filename:
        filename_no_ext, ext = os.path.splitext(results_filename)
        if ext and ext in _FMT_EXT.values():
            # if filename provided with one of expected ext,
            # remove it as it will be added back
            results_filename = filename_no_ext
    else:
        # base default filename on model name + img-size
        img_size = data_config["input_size"][1]
        results_filename = f'{args.model}-{img_size}'

    if args.results_dir:
        results_filename = os.path.join(args.results_dir, results_filename)

    for fmt in args.results_format:
        save_results(df, results_filename, fmt)

    print(f'--result')
    print(df.set_index(args.filename_col).to_json(orient='index', indent=4))


def save_results(df, results_filename, results_format='csv', filename_col='filename'):
    results_filename += _FMT_EXT[results_format]
    if results_format == 'parquet':
        df.set_index(filename_col).to_parquet(results_filename)
    elif results_format == 'json':
        df.set_index(filename_col).to_json(results_filename, indent=4, orient='index')
    elif results_format == 'json-records':
        df.to_json(results_filename, lines=True, orient='records')
    elif results_format == 'json-split':
        df.to_json(results_filename, indent=4, orient='split', index=False)
    else:
        df.to_csv(results_filename, index=False)


if __name__ == '__main__':
    main()
