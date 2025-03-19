import torch.nn as nn
import torch.optim as optim
from modelling.vtn_att_poseflow_model import VTNHCPF_GCN
import torch
from trainer.tools import MyCustomLoss,OLM_Loss
from collections import OrderedDict
from pytorch_lightning.utilities.migration import pl_legacy_patch

def load_criterion(train_cfg):
    criterion = None
    if train_cfg['criterion'] == "MyCustomLoss":
        criterion = MyCustomLoss(label_smoothing=train_cfg['label_smoothing'])
    if train_cfg['criterion'] == "OLM_Loss": 
        criterion = OLM_Loss(label_smoothing=train_cfg['label_smoothing'])
    assert criterion is not None
    return criterion

def load_optimizer(train_cfg,model):
    optimzer = None
    if train_cfg['optimzer'] == "SGD":
        optimzer = optim.SGD(model.parameters(), lr=train_cfg['learning_rate'],weight_decay=float(train_cfg['w_decay']),momentum=0.9,nesterov=True)
    if train_cfg['optimzer'] == "Adam":
        optimzer = optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'],weight_decay=float(train_cfg['w_decay']))
    assert optimzer is not None
    return optimzer

def load_lr_scheduler(train_cfg,optimizer):
    scheduler = None
    if train_cfg['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=train_cfg['scheduler_factor'], patience=train_cfg['scheduler_patience'])
    if train_cfg['lr_scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_step_size'], gamma=train_cfg['gamma'])
    assert scheduler is not None
    return scheduler

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    try:
        if m.weight is not None:
            m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    except:
        pass


def load_model(cfg):
    if cfg['training']['pretrained']:
        print(f"load pretrained model: {cfg['training']['pretrained_model']}")
        if cfg['data']['model_name'] == 'VTNGCN':
            model = VTNHCPF_GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            if '.ckpt' in cfg['training']['pretrained_model']: #Temporary
                new_state_dict = {}
                with pl_legacy_patch():
                    checkpoint = torch.load(cfg['training']['pretrained_model'], map_location='cpu')['state_dict']
                    for key, value in checkpoint.items():
                        new_key = key.replace('model.', '')
                        if not new_key.startswith('bottle_mm') and not new_key.startswith('self_attention_decoder') and not new_key.startswith('classifier'):
                        # if not new_key.startswith('bottle_mm') and not new_key.startswith('self_attention_decoder'):
                        # if not new_key.startswith('bottle_mm') and not new_key.startswith('classifier'):
                        # if not new_key.startswith('classifier'):
                            new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict, strict=False)
                model.reset_head(model.num_classes)
            else:
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
    else:
        if cfg['data']['model_name'] == 'VTNGCN':
            model = VTNHCPF_GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])

    assert model is not None
    print("loaded model")
    return model
        