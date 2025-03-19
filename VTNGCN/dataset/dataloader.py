from dataset.dataset import build_dataset
import torch

def vtn_gcn_collate_fn_(batch):
    clip = torch.stack([s[0] for s in batch], dim = 0)
    poseflow = torch.stack([s[1] for s in batch], dim = 0)
    keypoints = torch.stack([s[2] for s in batch], dim = 0)
    labels = torch.stack([s[3] for s in batch], dim = 0)
    return {'clip':clip, 'poseflow':poseflow, 'keypoints':keypoints},labels


def build_dataloader(cfg, split, is_train=True, model = None,labels = None):
    dataset = build_dataset(cfg['data'], split,model,train_labels = labels)
    if cfg['data']['model_name'] == 'VTNGCN' or cfg['data']['model_name'] == 'VTNGCN_Combine':
        collate_func = vtn_gcn_collate_fn_

    dataloader = torch.utils.data.DataLoader(dataset,
                                            collate_fn = collate_func,
                                            batch_size = cfg['training']['batch_size'],
                                            num_workers = cfg['training'].get('num_workers',2),
                                            shuffle = is_train,
                                            prefetch_factor = cfg['training'].get('prefetch_factor',2),
                                            pin_memory=True,
                                            persistent_workers =  True,
                                            # sampler = sampler
                                            )

    return dataloader
