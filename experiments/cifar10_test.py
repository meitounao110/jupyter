import sys
import torch
sys.path.append("..")
import main
from util.Config import parse_dict_args


def parameters():
    # 定义默认参数
    defaults = {
        # Technical details
        'is_parallel': False,
        'workers': 2,
        'gpu': 0,
        'checkpoint_epochs': 20,

        # Data
        'dataset': 'FPN',
        'base_batch_size': 128,
        'base_labeled_batch_size': 64,
        'print_freq': 1,
        'train_subdir': 'train/train',
        'eval_subdir': 'test/0/test',
        'mean_layout': 0,
        'std_layout': 20000,
        'list_path1': '/mnt/share1/layout_data/v0.3/data/all_walls/train/train.txt',  # 放无标签数据列表
        'list_path2': '/mnt/share1/layout_data/v0.3/data/all_walls/train/val.txt',  #有标签
        'test_list_path': '/mnt/share1/layout_data/v0.3/data/all_walls/test/test_0.txt',  # 存放测试样本

        # Architecture
        'arch': 'FPN1',
        # 'arch': 'lenet',
        # 'arch': 'vgg19',
        # 'arch': 'resnet18',
        # 'arch': 'preact_resnet18',
        # 'arch': 'densenet121',
        # 'arch': 'resnext29_32x4d',
        # 'arch': 'senet',
        # 'arch': 'dpn92',
        # 'arch': 'shuffleG3',
        # 'arch': 'mobileV2',

        # Optimization
        'loss': 'mse',
        'optim': 'adam',
        'epochs': 100,
        'base_lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'nesterov': True,

        # LR_schedular
        # 'lr_scheduler': 'none',
        'lr_scheduler': 'cos',
        # 'lr_scheduler': 'multistep',
        'steps': '100,150,200,250,300,350,400,450,480',
        'gamma': 0.5,
        'min_lr': 1e-4,

        # Pseudo-Label
        't1': 10,
        't2': 60,
        'af': 0.3,
        'n_labels': 4000,
        'data_seed': 10,
        'upsize': 200,
    }

    return defaults


def run(base_batch_size, base_labeled_batch_size, base_lr, n_labels, data_seed, is_parallel, **kwargs):
    if is_parallel and torch.cuda.is_available():
        ngpu = torch.cuda.device_count()
    else:
        ngpu = 1
    adapted_args = {
        'batch_size': base_batch_size * ngpu,
        'labeled_batch_size': base_labeled_batch_size * ngpu,
        'lr': base_lr,
        # 'labels': '../data-local/labels/cifar10/{}_balanced_labels/{:02d}.txt'.format(n_labels, data_seed),
        'labels': '/mnt/share1/layout_data/v0.3/data/all_walls/train/val.txt',
        'is_parallel': is_parallel,
    }
    args = parse_dict_args(**adapted_args, **kwargs)
    print('args', args)
    main.main(args)


if __name__ == "__main__":
    run_params = parameters()
    # 参数前双星视为字典
    run(**run_params)
