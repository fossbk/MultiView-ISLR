import torch
from torch import nn

from .vtn_utils import FeatureExtractor, FeatureExtractorGCN, LinearClassifier, SelfAttention, CrossAttention
import torch.nn.functional as F
from pytorch_lightning.utilities.migration import pl_legacy_patch

class MMTensorNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, x):
        mean = torch.mean(x, dim=self.dim).unsqueeze(self.dim)
        std = torch.std(x, dim=self.dim).unsqueeze(self.dim)
        return (x - mean) / std

class VTNHCPF_GCN(nn.Module):
    def __init__(self, num_classes=199, num_heads=4, num_layers=2, embed_size=512, sequence_length=16, cnn='rn34',
                 gcn='AAGCN', freeze_layers=0, dropout=0, **kwargs):
        super().__init__()
        print("Model: VTNHCPF_GCN")
        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes

        self.feature_extractor = FeatureExtractor(cnn, embed_size, freeze_layers)
        self.feature_extractor_gcn = FeatureExtractorGCN(gcn, freeze_layers)

        num_attn_features = embed_size
        num_gcn_features = int(embed_size/2)
        pose_flow_features = 106
        # pose_flow_features = 0
        add_attn_features = 0
        self.norm = MMTensorNorm(-1)
        self.bottle_mm = nn.Linear(pose_flow_features + num_attn_features + num_gcn_features, num_attn_features + add_attn_features)

        self.self_attention_decoder = SelfAttention(num_attn_features + add_attn_features, num_attn_features + add_attn_features,
                                                    [num_heads] * num_layers,
                                                    self.sequence_length, layer_norm=True,dropout = dropout)
        self.classifier = LinearClassifier(num_attn_features + add_attn_features, num_classes, dropout)
        self.num_attn_features  = num_attn_features + add_attn_features
        self.dropout = dropout
        self.num_classes = num_classes
        self.relu = F.relu

    def reset_head(self,num_classes):
        self.classifier = LinearClassifier(self.num_attn_features, num_classes, self.dropout)
        print("Reset to ",num_classes)

    def forward_features(self, features = None,poseflow = None, features_keypoint = None):

        # Reshape to put both hand crops on the same axis.
       
        zp = torch.cat((features,features_keypoint,poseflow), dim=-1)

        zp = self.norm(zp)
        zp = self.relu(self.bottle_mm(zp))

        zp = self.self_attention_decoder(zp)

        return zp
    
    def forward(self, clip = None,poseflow = None,keypoints = None,**kwargs):
        """Extract the image feature vectors."""
        rgb_clip, pose_clip = clip,poseflow

        # Reshape to put both hand crops on the same axis.
        b, t, x, c, h, w = rgb_clip.size()
        rgb_clip = rgb_clip.view(b, t * x, c, h, w)
        zc = self.feature_extractor(rgb_clip)
        # Reshape back to extract features of both wrist crops as one feature vector.
        zc = zc.view(b, t, -1)    
        zk = self.feature_extractor_gcn(keypoints)

        zp = torch.cat((zc,pose_clip,zk), dim=-1)

        zp = self.norm(zp)
        zp = self.relu(self.bottle_mm(zp))

        zp = self.self_attention_decoder(zp)

        y = self.classifier(zp)

        return {'logits':y.mean(1)} # train
        # return y.mean(1) # convert to script

