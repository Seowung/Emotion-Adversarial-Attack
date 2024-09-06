import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torchvision import models
from torch.nn.utils.weight_norm import weight_norm

import sys
from abc import abstractmethod

sys.path.append('../')
from util import ClassWisePool
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class WSCNet(nn.Module):

    def __init__(self, num_classes=8, num_maps=8):
        super(WSCNet, self).__init__()
        model = models.resnet101(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.num_ftrs = model.fc.in_features

        self.downconv = nn.Sequential(
            nn.Conv2d(2048, num_classes * num_maps, kernel_size=1, stride=1, padding=0, bias=True))

        self.GAP = nn.AvgPool2d(14)
        self.GMP = nn.MaxPool2d(14)

        pooling = nn.Sequential()
        pooling.add_module('class_wise', ClassWisePool(num_maps))
        pooling2 = nn.Sequential()
        pooling2.add_module('class_wise', ClassWisePool(num_classes))

        self.spatial_pooling = pooling
        self.spatial_pooling2 = pooling2
        self.classifier = nn.Sequential(
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x_ori = x
        # detect branch
        x = self.downconv(x)
        x_conv = x
        x = self.GAP(x)  # x = self.GMP(x)
        x = self.spatial_pooling(x)
        x = x.view(x.size(0), -1)
        
        # cls branch
        x_conv = self.spatial_pooling(x_conv)
        x_conv = x_conv * x.view(x.size(0), x.size(1), 1, 1)
        x_conv = self.spatial_pooling2(x_conv)
        x_conv_copy = x_conv
        for num in range(0, 2047):
            x_conv_copy = torch.cat((x_conv_copy, x_conv), 1)
        x_conv_copy = torch.mul(x_conv_copy, x_ori)
        x_conv_copy = torch.cat((x_ori, x_conv_copy), 1)
        x_conv_copy = self.GAP(x_conv_copy)
        x_conv_copy = x_conv_copy.view(x_conv_copy.size(0), -1)
        x_conv_copy = self.classifier(x_conv_copy)
        
        return x_conv_copy
        #return x, x_conv_copy


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class CAERSNet(BaseModel):
    def __init__(self, use_face=False, use_context=True, concat=False):
        super().__init__()
        self.two_stream_net = TwoStreamNetwork()
        self.fusion_net = FusionNetwork(use_face, use_context, concat)

    def forward(self, face=None, context=None):
        face, context = self.two_stream_net(face, context)

        return self.fusion_net(face, context)


class Encoder(nn.Module):
    def __init__(self, num_kernels, kernel_size=3, bn=True, max_pool=True, maxpool_kernel_size=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        n = len(num_kernels) - 1
        self.convs = nn.ModuleList([nn.Conv2d(
            num_kernels[i], num_kernels[i + 1], kernel_size, padding=padding) for i in range(n)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(num_kernels[i + 1])
                                 for i in range(n)]) if bn else None
        self.max_pool = nn.MaxPool2d(maxpool_kernel_size) if max_pool else None

    def forward(self, x):
        n = len(self.convs)
        for i in range(n):
            x = self.convs[i](x)
            if self.bn is not None:
                x = self.bn[i](x)
            x = F.relu(x)
            if self.max_pool is not None and i < n - 1:  # check if i < n
                x = self.max_pool(x)
        return x


class TwoStreamNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        num_kernels = [3, 32, 64, 128, 256, 256]
        self.face_encoding_module = Encoder(num_kernels)
        self.context_encoding_module = Encoder(num_kernels)
        self.attention_inference_module = Encoder(
            [256, 128, 1], max_pool=False)

    def forward(self, face, context):
        face = self.face_encoding_module(face)

        context = self.context_encoding_module(context)
        attention = self.attention_inference_module(context)
        N, C, H, W = attention.shape
        attention = F.softmax(attention.view(
            N, -1), dim=-1).view(N, C, H, W)
        context = context * attention

        return face, context


class FusionNetwork(nn.Module):
    def __init__(self, use_face=True, use_context=True, concat=False, num_class=8):
        super().__init__()
        # add batch norm to ensure the mean and std of
        # face and context features are not too different
        self.face_bn = nn.BatchNorm1d(256)
        self.context_bn = nn.BatchNorm1d(256)

        self.use_face, self.use_context = use_face, use_context
        self.concat = concat

        self.face_1 = nn.Linear(256, 128)
        self.face_2 = nn.Linear(128, 1)

        self.context_1 = nn.Linear(256, 128)
        self.context_2 = nn.Linear(128, 1)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_class)

        self.dropout = nn.Dropout()

    def forward(self, face, context):
        face = F.avg_pool2d(face, face.shape[2]).view(face.shape[0], -1)
        context = F.avg_pool2d(context, context.shape[2]).view(context.shape[0], -1)

        # add batch norm for face and context branch
        face, context = self.face_bn(face), self.context_bn(context)

        if not self.concat:
            lambda_f = F.relu(self.face_1(face))
            lambda_c = F.relu(self.context_1(context))

            lambda_f = self.face_2(lambda_f)
            lambda_c = self.context_2(lambda_c)

            weights = torch.cat([lambda_f, lambda_c], dim=-1)
            weights = F.softmax(weights, dim=-1)
            face = face * weights[:, 0].unsqueeze(dim=-1)
            context = context * weights[:, 1].unsqueeze(dim=-1)

        if not self.use_face:
            face = torch.zeros_like(face)

        if not self.use_context:
            context = torch.zeros_like(context)

        features = torch.cat([face, context], dim=-1)
        features = F.relu(self.fc1(features))
        features = self.dropout(features)

        return self.fc2(features)


class PDANet(nn.Module):
    def __init__(self, C=8, activate_type="none"):
        super(PDANet, self).__init__()
        self.base = models.resnet101(pretrained=True)
        self.se_block = None
        self.se_block = SENet_block(activate_type)
        self.fc = nn.Linear(2048 * 2, 3)
        self.fc_classify = nn.Linear(2048 * 2, C)
        self.spatial = spatial_block()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if (name == 'avgpool'):
                break
            x = module(x)
        out, out_channel_wise, out2 = self.se_block(x)
        out = torch.mean(out.view(-1, out.size(1), out.size(2) * out.size(3)), 2)
        spatial_feature = self.spatial(out2, out_channel_wise)
        feature_cat = torch.cat((out, spatial_feature), 1)
        out = self.fc(feature_cat)
        out_classify = self.fc_classify(feature_cat)
        return out_classify
        #return out, out_classify, out


class SENet_block(nn.Module):
    def __init__(self, activate_type="none"):
        super(SENet_block, self).__init__()
        self.conv1 = nn.Conv2d(2048, 2048, 1)
        self.conv2 = nn.Conv2d(2048, 2048, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv_classify = nn.Conv2d(2048, 2048, 1)
        self.activate_type = activate_type

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.sigmoid(out1)
        out1 = torch.mean(out1.view(-1, out1.size(1), out1.size(2) * out1.size(3)), 2)
        out_channel_wise = None
        if self.activate_type == "none":
            out_channel_wise = out1
        if self.activate_type == "softmax":
            out_channel_wise = F.softmax(out1, dim=1)
        if self.activate_type == "sigmoid":
            out_channel_wise = F.sigmoid(out1)
        if self.activate_type == "sigmoid_res":
            out_channel_wise = F.sigmoid(out1) + 1
        out2 = self.conv2(x)
        out2 = self.relu(out2)
        out = out2 * out_channel_wise.view(-1, out1.size(1), 1, 1)
        return out, out_channel_wise, out2


class spatial_block(nn.Module):
    def __init__(self):
        super(spatial_block, self).__init__()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(2048, 2048)
        self.conv = nn.Conv2d(2048, 2048, 1)
        self.conv2 = nn.Conv2d(2048, 1, 1)

    def forward(self, x, channel_wise):
        out = self.conv(x)
        if len(channel_wise.size()) != len(x.size()):
            channel_wise = self.fc(channel_wise)
            channel_wise = channel_wise.view(-1, channel_wise.size(1), 1, 1)
            out = out + channel_wise
        out = self.tanh(out)
        out = self.conv2(out)
        x_shape = out.size(2)
        y_shape = out.size(3)
        out = out.view(-1, x_shape * y_shape)
        out = F.softmax(out, dim=1)
        out = out.view(-1, 1, x_shape, y_shape)
        out = x * out
        out = torch.mean(out.view(-1, out.size(1), out.size(2) * out.size(3)), 2)
        return out

class Stimuli_Aware_VEA(nn.Module):
    """ResNet50 for Visual Sentiment Analysis on FI_8"""
    # """ResNet50 for Visual Sentiment Analysis on flickr_2"""

    def __init__(self, base_model=models.resnet50(pretrained=True)):
        super(Stimuli_Aware_VEA, self).__init__()
        self.fcn = nn.Sequential(*list(base_model.children())[:-2])

        self.face = models.resnet18()
        self.conv1 = nn.Conv1d(3072, 3072, 1, bias=True)
        self.lstm = Decoder(feat_size=2048, hidden_size=512)
        self.sigmoid = nn.Sigmoid()

        self.GAvgPool = nn.AvgPool2d(kernel_size=14)
        self.classifiers_x = nn.Sequential(
            nn.Linear(in_features=2048, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=1)
            )

        self.classifier8 = nn.Sequential(
            nn.Linear(in_features=2048 + 512 +512, out_features=8)
            )

        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, face, rcnn, fmask):
        x1 = self.fcn(x)
        x = self.GAvgPool(x1)
        x = x.view(x.size(0), x.size(1))

        face, face1 = self.face(face)
        rcnn, _, alpha = self.lstm(rcnn)

        #-------classifier8--------#
        features = torch.cat([x, rcnn, face], dim=1)
        emotion = self.classifier8(features)

        #-------8to2-------#s
        emotion = F.softmax(emotion, dim=1)

        positive = emotion[:, 0:4].sum(1)
        negative = emotion[:, 4:8].sum(1)

        positive = positive.view(positive.size(0), 1)
        negative = negative.view(negative.size(0), 1)

        sentiment = torch.cat([positive, negative], dim=1)
        
        return emotion

        #return emotion, sentiment


class Attention(nn.Module):
    def __init__(self, feat_size, hidden_size, att_size, dropout=0.5):
        super(Attention, self).__init__()

        self.feats_att = weight_norm(nn.Linear(feat_size, att_size))
        self.hiddens_att = weight_norm(nn.Linear(hidden_size, att_size))
        self.full_att = weight_norm(nn.Linear(att_size, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feats, hiddens):
        att1 = self.feats_att(feats)
        att2 = self.hiddens_att(hiddens)

        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)
        alpha = self.softmax(att)
        atted_feats = (feats * alpha.unsqueeze(2)).sum(dim=1)

        return atted_feats, alpha


class Decoder(nn.Module):
    def __init__(self, feat_size, hidden_size):
        super(Decoder, self).__init__()

        self.minval = -0.1
        self.maxval = 0.1
        self.fc_size = 1024
        self.dropout = 0.5
        self.batch_size = None
        self.hidden_size = hidden_size
        self.feat_size = feat_size
        self.maxlen = None

        self.LSTM1 = nn.LSTMCell(self.feat_size + self.hidden_size, self.hidden_size, bias=True)
        self.LSTM2 = nn.LSTMCell(self.feat_size + self.hidden_size, self.hidden_size, bias=True)
        self.Att = Attention(self.feat_size, self.hidden_size, self.hidden_size)

        self.weights_init()

    def forward(self, feats):
        self.batch_size = feats.shape[0]
        h1, c1 = self.init_hidden_state(self.batch_size)
        h2, c2 = self.init_hidden_state(self.batch_size)
        mean_feat = feats.mean(1)
        thought_vectors = torch.zeros(self.batch_size, 3, self.hidden_size)
        alpha_mat = torch.zeros(self.batch_size, 10)
        for t in range(1):
            h1, c1 = self.LSTM1(torch.cat([mean_feat, h2], dim=1), (h1, c1))
            att_feats, alpha = self.Att(feats, h1)
            alpha_mat = alpha
            h2, c2 = self.LSTM2(torch.cat([att_feats, h1], dim=1), (h2, c2))
            thought_vectors[:, t, :] = h2


        return h2, h1, alpha_mat

    def init_hidden_state(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_size)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.hidden_size)
        return h, c

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight.data, a=self.minval, b=self.maxval)
                m.bias.data.fill_(0)


model = Stimuli_Aware_VEA()
print(model)
