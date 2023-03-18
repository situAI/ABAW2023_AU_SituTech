import torch
import timm
import timm.scheduler
import copy
import inspect
from core.utils import OPTIMIZER_REGISTRY, LR_SCHEDULER_REGISTRY

def register_torch_optimizers():
    """
    Register all optimizers implemented by torch
    """
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZER_REGISTRY.register()(_optim)


def build_optimizer(model_params, cfg):
    register_torch_optimizers()

    return OPTIMIZER_REGISTRY.get(cfg['name'])(model_params, **cfg['args'])


def register_torch_lr_scheduler():
    """
    Register all lr_schedulers implemented by torch
    """
    for module_name in dir(torch.optim.lr_scheduler):
        if module_name.startswith('__'):
            continue
        
        _scheduler = getattr(torch.optim.lr_scheduler, module_name)
        if inspect.isclass(_scheduler) and issubclass(_scheduler, torch.optim.lr_scheduler._LRScheduler):
            LR_SCHEDULER_REGISTRY.register()(_scheduler)


def register_timm_lr_scheduler():
    """
    Register all lr_schedulers implemented by timm
    """
    for module_name in dir(timm.scheduler):
        if module_name.startswith('__') or 'create' in module_name:
            continue

        _scheduler = getattr(timm.scheduler, module_name)
        if inspect.isclass(_scheduler) and issubclass(_scheduler, timm.scheduler.scheduler.Scheduler):
            LR_SCHEDULER_REGISTRY.register()(_scheduler)


def build_lr_scheduler(optimizer, cfg):
    register_timm_lr_scheduler()
    register_torch_lr_scheduler()

    return LR_SCHEDULER_REGISTRY.get(cfg['name'])(optimizer, **cfg['args'])

