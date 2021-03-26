import sys
import torch

sys.path.append("..")
import main
from util.Config import parse_dict_args


def parameters():
    # 定义默认参数
    defaults = {
        # Technical details
        'train': True,
        'is_parallel': False,
        'workers': 8,
        'gpu': 2,
        'checkpoint_epochs': 20,

        # Data
        'dataset': 'FPN',
        'base_batch_size': 8,
        'print_freq': 20,
        'train_subdir': "train",
        # 'one_point/train/train',
        'eval_subdir': 'train',
        # 'one_point/test/0/test',
        'mean_layout': 0,
        'std_layout': 10000,
        'list_path1':  # None,                           # 放无标签数据列表
            '/mnt/zhangyunyang1/pseudo_label-pytorch-master/4001-8000label.txt',
        'list_path2':   None,                              # 有标签
            #'/mnt/zhangyunyang1/pseudo_label-pytorch-master/500label.txt',
        'test_list_path':  # '/mnt/zhangyunyang1/pseudo_label-pytorch-master/1600label.txt',
            '/mnt/zhaoxiaoyu/data/layout_data/simple_component/dataset/200x200_val.txt',  # 存放测试样本
        'PATH': '/mnt/zhangyunyang1/pseudo_label-pytorch-master/experiments/model/uns_onepoint200_7900ul.pth',
        # modlePATH

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
        'epochs': 200,
        'base_lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'nesterov': True,

        # LR_schedular
        'lr_scheduler': 'none',
        # 'lr_scheduler': 'cos',
        # 'lr_scheduler': 'multistep',
        'steps': '100,150,200,250,300,350,400,450,480',
        'gamma': 0.5,
        'min_lr': 1e-4,

        # Pseudo-Label
        't1': 10,
        't2': 60,
        'af': 1,
        'upsize': 200,
    }

    return defaults


def run(base_batch_size, base_lr, is_parallel, **kwargs):
    if is_parallel and torch.cuda.is_available():
        ngpu = torch.cuda.device_count()
    else:
        ngpu = 1
    adapted_args = {
        'batch_size': base_batch_size * ngpu,
        'lr': base_lr,
        'is_parallel': is_parallel,
    }
    args = parse_dict_args(**adapted_args, **kwargs)
    print('args', args)
    main.main(args)


if __name__ == "__main__":
    run_params = parameters()
    # 参数前双星视为字典
    run(**run_params)
