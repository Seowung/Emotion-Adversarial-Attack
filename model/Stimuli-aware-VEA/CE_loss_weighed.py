import numpy as np
import torch

class CELoss_weighed():

    def __init__(self, weight=None, size_average=True):
        """
        初始化参数，因为要实现 torch.nn.CrossEntropyLoss 的两个比较重要的参数

        :param weight: 给予每个类别不同的权重
        :param size_average: 是否要对 loss 求平均
        """

        self.weight = weight
        self.size_average = size_average


    def __call__(self, emo_vec, senti_vec, emo, senti):
        """
        计算损失
        这个方法让类的实例表现的像函数一样，像函数一样可以调用

        :param input: (batch_size, C)，C是类别的总数
        :param target: (batch_size, 1)
        :return: 损失
        """

        batch_loss = 0.
        for i in range(emo_vec.shape[0]):

            # 计算单个损失
            loss1 = -torch.log(emo_vec[i, emo[i]])
            # loss2 = -torch.log(senti_vec[i, senti[i]])
            loss2 = 1 - senti_vec[i, senti[i]]

            loss = loss1 + loss1 * loss2
            # loss = loss1 + loss2

            # 损失累加
            batch_loss += loss

        batch_loss /= emo.shape[0]

        return batch_loss
