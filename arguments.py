import argparse
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Building wireframe Transformer', add_help=False)
    parser.add_argument('--exp_dataset', default='outdoor',
                        help='the dataset for experiments')
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='base learning rate for corner/edge heads')
    parser.add_argument('--lr_backbone', default=1e-4, type=float,
                        help='learning rate for the ResNet backbone; set <=0 to reuse --lr')
    parser.add_argument('--lr_roof', default=8e-4, type=float,
                        help='learning rate for the roof prior auxiliary head; set <=0 to reuse --lr')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=220, type=int)
    parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine', 'step'],
                        help='learning-rate schedule; cosine uses warmup_epochs and min_lr_ratio')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='number of warmup epochs for cosine schedule')
    parser.add_argument('--min_lr_ratio', default=0.05, type=float,
                        help='final LR as a ratio of each param group base LR for cosine schedule')
    parser.add_argument('--lr_drop', default=160, type=int,
                        help='step size for --lr_scheduler step; kept for compatibility')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--print_freq', default=40, type=int)
    parser.add_argument('--output_dir', default='./checkpoints/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./tensorboard',
                        help='directory for TensorBoard event files')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--load_weights', default='',
                        help='initialize model weights from checkpoint without restoring optimizer/scheduler state')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--val_batch_size', default=8, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--max_corner_num', default=150, type=int,
                        help='the max number of corners allowed in the experiments')
    parser.add_argument('--corner_limit', default=150, type=int,
                        help='skip training/validation samples whose ground-truth corner count exceeds this limit; set <=0 to disable')
    parser.add_argument('--corner_to_edge_multiplier', default=3, type=int,
                        help='the max number of edges based on the number of corner candidates (assuming the '
                             'average degree never greater than 6)')
    parser.add_argument('--lambda_corner', default=0.05, type=float,
                        help='the max number of corners allowed in the experiments')
    parser.add_argument('--lambda_roof', default=0.15, type=float,
                        help='weight for the roof structure prior auxiliary loss; set 0 to disable its effect')
    parser.add_argument('--freeze_backbone_epochs', default=0, type=int,
                        help='freeze backbone parameters for the first N epochs during finetuning')
    parser.add_argument('--save_every', default=20, type=int,
                        help='save an extra numbered checkpoint every N epochs')
    parser.add_argument('--val_every', default=10, type=int,
                        help='run validation every N epochs when --run_validation is set; set <=0 to disable validation')
    parser.add_argument('--max_train_samples', default=0, type=int,
                        help='limit the number of training samples per epoch; set <=0 to use all samples')
    parser.add_argument('--max_val_samples', default=0, type=int,
                        help='limit the number of validation samples; set <=0 to use all samples')
    parser.add_argument('--sample_seed', default=42, type=int,
                        help='deterministic seed used when limiting train/validation samples')
    parser.add_argument('--data_path', default='./building3d/b3d',
                        help='processed dataset root containing rgb/ annot/ and list files')
    parser.add_argument('--train_list', default='',
                        help='optional explicit training split file')
    parser.add_argument('--val_list', default='',
                        help='optional explicit validation split file')
    parser.add_argument('--test_list', default='',
                        help='optional explicit test split file')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training')
    parser.add_argument('--run_validation', action='store_true',
            help='Whether run validation or not, default: False')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    return parser
