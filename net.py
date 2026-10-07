from torch import nn
import os
from loss import SoftIoULoss
from model.FSGPNet.FSGPNet import FSGPNet


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Net(nn.Module):
    def __init__(self, model_name, mode='test'):
        super(Net, self).__init__()

        self.model_name = model_name
        self.cal_loss = SoftIoULoss()

        if model_name == 'FSGPNet':
            self.model = FSGPNet()
        else:
            raise ValueError(f"Unsupported model: {model_name}. Choose FSGPNet.")


    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
