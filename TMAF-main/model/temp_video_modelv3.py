import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1):
        super(ConvBlock, self).__init__()

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding="same")
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        # Shortcut layer needs to match the output dimensions of the conv1d output
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels)
        )
        self.relu2 = nn.ReLU()
    def forward(self, x):
        # Perform convolution and batch normalization
        out = self.conv1d(x)
        out = self.bn(out)
        

        # Apply shortcut and add to output
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu2(out)  # Apply ReLU activation after shortcut addition

        return out


class SlidingWindowModel(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, kernel_size=2, stride=1):
        super(SlidingWindowModel, self).__init__()

        layers = []
        for i in range(num_blocks):
            layers.append(ConvBlock(in_channels, out_channels, kernel_size, stride))
            in_channels = out_channels  # Update in_channels for the next block

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class FuseModel(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, kernel_size=2, stride=1):
        super(FuseModel, self).__init__()

        layers = []
        for i in range(num_blocks):
            layers.append(ConvBlock(in_channels, out_channels, kernel_size, stride))
            in_channels = out_channels  # Update in_channels for the next block

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ExpendAs(nn.Module):
    def __init__(self, rep):
        super(ExpendAs, self).__init__()
        self.rep = rep

    def forward(self, tensor):
        return tensor.repeat(1, self.rep, 1)


# Channel attention
class ChannelBlock(nn.Module):
    def __init__(self, in_channels, filter_size):
        super(ChannelBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, filter_size, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm1d(filter_size)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv1d(in_channels, filter_size, kernel_size=5, padding=2)
        self.batch2 = nn.BatchNorm1d(filter_size)
        self.relu2 = nn.ReLU(inplace=False)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(filter_size * 2, filter_size)
        self.batch3 = nn.BatchNorm1d(filter_size)
        self.relu3 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(filter_size, filter_size)

        self.conv3 = nn.Conv1d(filter_size * 2, filter_size, kernel_size=1, padding="same")
        self.batch4 = nn.BatchNorm1d(filter_size)
        self.relu4 = nn.ReLU(inplace=False)

    def forward(self, x_2s, x_5s):
        conv1 = self.conv1(x_2s)
        batch1 = self.batch1(conv1)
        relu1 = self.relu1(batch1)

        conv2 = self.conv2(x_5s)
        batch2 = self.batch2(conv2)
        relu2 = self.relu2(batch2)

        concat = torch.cat([relu1, relu2], dim=1)

        pooled = self.global_pool(concat)
        pooled = pooled.view(pooled.shape[0], -1)
        fc1 = self.fc1(pooled)

       
        relu3 = self.relu3(fc1)
        fc2 = self.fc2(relu3)
        sig = torch.sigmoid(fc2)
        a = sig.view(sig.size(0), sig.size(1), 1)

        np.savetxt("channala.txt", a.detach().cpu().numpy().reshape(-1), fmt="%.6f")

        a1 = 1 - sig
        a1 = a1.view(a1.size(0), a1.size(1), 1)

        np.savetxt("channal1-a.txt", a1.detach().cpu().numpy().reshape(-1), fmt="%.6f")

        y = relu1 * a
        y1 = relu2 * a1

        concat_y_y1 = torch.cat([y, y1], dim=1)
        conv3 = self.conv3(concat_y_y1)
        batch4 = self.batch4(conv3)
        relu4 = self.relu4(batch4)
        return relu4


# Spatial attention
class SpatialBlock(nn.Module):
    def __init__(self, in_channels, filter_size, size):
        super(SpatialBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, filter_size, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm1d(filter_size)
        self.relu1 = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size=1, padding="same")
        self.batch2 = nn.BatchNorm1d(filter_size)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv1d(filter_size, 1, kernel_size=1, padding="same") 

        self.expend_as = ExpendAs(filter_size)  
        self.conv4 = nn.Conv1d(filter_size * 2, filter_size, kernel_size=size, padding="same")
        self.batch4 = nn.BatchNorm1d(filter_size)

    def forward(self, x, channel_data, filter_size, size):
        conv1 = self.conv1(x)
        batch1 = self.batch1(conv1)
        relu1 = self.relu1(batch1)

        conv2 = self.conv2(relu1)
        batch2 = self.batch2(conv2)
        relu2 = self.relu2(batch2)

        spatil_data = relu2

        data3 = F.relu(channel_data + spatil_data)
        data3 = self.conv3(data3)
        data3 = torch.sigmoid(data3)
        np.savetxt("Spatiala.txt", data3.detach().cpu().numpy().reshape(-1), fmt="%.6f")
        a = self.expend_as(data3)

        y = a * channel_data

        a1 = 1 - data3
        np.savetxt("Spatial1-a.txt", a1.detach().cpu().numpy().reshape(-1), fmt="%.6f")
        a1 = self.expend_as(a1)
        y1 = a1 * spatil_data

        data_a_a1 = torch.cat([y, y1], dim=1)

        conv3 = self.conv4(data_a_a1)
        batch3 = self.batch4(conv3)
        return batch3


class HAAM(nn.Module):
    def __init__(self, in_channels, filter_size, size):
        super(HAAM, self).__init__()
        self.channel_block = ChannelBlock(in_channels, filter_size)
        self.spatial_block = SpatialBlock(in_channels, filter_size, size)
        self.fc = nn.Conv1d(filter_size, 10, kernel_size=1, stride=1)

    def forward(self, x_1s, x_2s, x_5s, filte, size):
        channel_data = self.channel_block(x_2s, x_5s)
        haam_data = self.spatial_block(x_1s, channel_data, filte, size)
        haam_data = self.fc(haam_data)
        return haam_data


class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
     
        self.relu = nn.ReLU(inplace=False)
        self.classifier = nn.Linear(d_model, 1)  # start and end
        self.event_classifier = nn.Linear(d_model, 28)
        
    def forward(self, fused_content):
        max_fused_content, _ = fused_content.max(1)
        logits = self.classifier(fused_content)
        
        class_logits = self.event_classifier(max_fused_content)
        class_scores = class_logits

        return logits, class_scores


class Encoder(Module):
    r"""Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        r"""Pass the input through the endocder layers in turn.

        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output)

        if self.norm:
            output = self.norm(output)

        return output


class Decoder(Module):
    r"""Decoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the DecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory):
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory)

        if self.norm:
            output = self.norm(output)

        return output


class EncoderLayer(Module):
    r"""EncoderLayer, which is borrowed from CMRAN.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
       
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src):
        r"""Pass the input through the endocder layer.
        """
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DecoderLayer(Module):
    r"""DecoderLayer, which is borrowed from CMRAN.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory):
        r"""Pass the inputs (and mask) through the decoder layer.
        """
        memory = torch.cat([memory, tgt], dim=0)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=False)
       

    def forward(self, feature):
    
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature


class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, query_feature, memory_feature):
        query_feature = self.affine_matrix(query_feature)
        output = self.decoder(query_feature, memory_feature)

        return output


class Temp_Model(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super(Temp_Model, self).__init__()
        self.video_txt_fc = nn.Conv1d(11,10, kernel_size=1)
        self.audio_txt_fc = nn.Conv1d(64*2 ,64, kernel_size=1)
        self.SlidingWindowModel1s = SlidingWindowModel(5, in_channels, in_channels, kernel_size=1)
        self.SlidingWindowModel2s = SlidingWindowModel(5, in_channels, in_channels, kernel_size=2)
        self.SlidingWindowModel5s = SlidingWindowModel(5, in_channels, in_channels, kernel_size=5)

        self.audio_SlidingWindowModel1s = SlidingWindowModel(5, 64, 64, kernel_size=1)
        self.audio_SlidingWindowModel2s = SlidingWindowModel(5, 64, 64, kernel_size=2)
        self.audio_SlidingWindowModel5s = SlidingWindowModel(5, 64, 64, kernel_size=5)

        self.haam = HAAM(in_channels, filter_size=64, size=3)
        self.audio_haam = HAAM(64, filter_size=64, size=3)
        self.video_encoder = InternalTemporalRelationModule(input_dim=768, d_model=256,
                                                            feedforward_dim=1024)
        self.video_decoder = CrossModalRelationAttModule(input_dim=768, d_model=256,
                                                         feedforward_dim=1024)

        self.audio_encoder = InternalTemporalRelationModule(input_dim=768, d_model=256,
                                                            feedforward_dim=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=768, d_model=256,
                                                         feedforward_dim=1024)
        self.video_fc = nn.Linear(256, out_features=28)
        self.audio_fc = nn.Linear(256, out_features=28)
        # self.out_put = nn.Linear(feature_dim, 28)
        self.localize_module = SupvLocalizeModule(256)
        self.vis_localize_module = SupvLocalizeModule(256)
        self.audio_localize_module = SupvLocalizeModule(256)
        self.fuse = FuseModel(5, 512, 256, kernel_size=3)

    def forward(self, feat, text_feat, audio_feat, audio_text_feat):
        feat = torch.cat([feat, text_feat],dim=1)
        feat = self.video_txt_fc(feat)
        audio_feat = torch.cat([audio_feat, audio_text_feat],dim=1)
        audio_feat = self.audio_txt_fc(audio_feat)
        feat1s = self.SlidingWindowModel1s(feat)
        feat2s = self.SlidingWindowModel2s(feat)
        feat5s = self.SlidingWindowModel5s(feat)
        #
        feat = self.haam(feat1s, feat2s, feat5s, filte=64, size=3)

        audio_feat1s = self.audio_SlidingWindowModel1s(audio_feat)
        audio_feat2s = self.audio_SlidingWindowModel2s(audio_feat)
        audio_feat5s = self.audio_SlidingWindowModel5s(audio_feat)
        #
        audio_feat = self.audio_haam(audio_feat1s, audio_feat2s, audio_feat5s, filte=64, size=3)

        vis_feat_encode = self.video_encoder(feat)
        audio_feat_encode = self.audio_encoder(audio_feat)
   
        B, f, C = audio_feat_encode.shape
        Fv = vis_feat_encode.reshape(B*f, C)
        Fa = audio_feat_encode.reshape(B*f, C)

        attn_scores = torch.matmul(Fv, Fa.T)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        audio_feat_encode = ((attn_scores @ Fa)).reshape(B,f,C) + audio_feat_encode
        #
        feat = torch.cat([vis_feat_encode, audio_feat_encode], dim=-1) 
        feat = self.fuse(feat.permute(0,2,1)).permute(0,2,1)
        # kl_loss
        video_fc = self.video_fc(vis_feat_encode)
        audio_fc = self.audio_fc(audio_feat_encode)
        video_fcc = nn.ReLU()(video_fc)
        audio_fcc = nn.ReLU()(audio_fc)
        video_sim = nn.Softmax(dim=-1)(video_fcc)
        audio_sim = nn.Softmax(dim=-1)(audio_fcc)
        kl_loss = F.kl_div(audio_sim.log(), video_sim, reduction='sum')

        is_event_scores, event_scores = self.localize_module(feat)
        vis_is_event_scores, vis_event_scores = self.vis_localize_module(vis_feat_encode)
        audio_is_event_scores, audio_event_scores = self.audio_localize_module(audio_feat_encode)

        return is_event_scores, event_scores, kl_loss, vis_is_event_scores, vis_event_scores, audio_is_event_scores, audio_event_scores






