import os
import math
import torch
import torch.nn.init as init
from torch.optim import lr_scheduler


def get_scheduler(optimizer, config, epochs=-1):
    if 'lr_policy' not in config or config.lr_policy == 'constant':
        scheduler = None
    elif config.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size,
                                        gamma=config.gamma, last_epoch=epochs)
    elif config.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.factor,
                                                    patience=config.patience, eps=1e-08)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config.lr_policy)
    return scheduler


def get_model_list(dirname, key):
    if not os.path.exists(dirname):
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    return gen_models[-1]


def get_model_ckpt_name(checkpoint_dir, key, epochs):
    if not os.path.exists(checkpoint_dir):
        return None
    ckpt_path = os.path.join(checkpoint_dir, key + f'_{epochs:04d}.pt')
    assert os.path.isfile(ckpt_path), f'Checkpoint not found: {ckpt_path}'
    return ckpt_path


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


def vgg_preprocess(batch):
    """Convert RGB batch from [-1,1] to BGR [0,255] with VGG mean subtraction."""
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)
    batch = (batch + 1) * 255 * 0.5
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(mean)
    return batch
