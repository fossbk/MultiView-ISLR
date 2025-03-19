from utils.video_augmentation import *
from dataset.vtn_att_poseflow_model_dataset import VTN_GCN_Dataset

def build_dataset(dataset_cfg, split,model = None,**kwargs):
    dataset = None

    if dataset_cfg['model_name'] == 'VTNGCN':
        dataset = VTN_GCN_Dataset(dataset_cfg['base_url'],split,dataset_cfg,**kwargs)

    assert dataset is not None
    return dataset



    