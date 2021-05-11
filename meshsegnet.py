import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.get_device())
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(x.get_device())
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class MeshSegNet(nn.Module):
    def __init__(self, num_classes=15, num_channels=15, with_dropout=True, dropout_p=0.5):
        super(MeshSegNet, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.with_dropout = with_dropout
        self.dropout_p = dropout_p

        # MLP-1 [64, 64]
        self.mlp1_conv1 = torch.nn.Conv1d(self.num_channels, 64, 1)
        self.mlp1_conv2 = torch.nn.Conv1d(64, 64, 1)
        self.mlp1_bn1 = nn.BatchNorm1d(64)
        self.mlp1_bn2 = nn.BatchNorm1d(64)
        # FTM (feature-transformer module)
        self.fstn = STNkd(k=64)
        # GLM-1 (graph-contrained learning modulus)
        self.glm1_conv1_1 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_conv1_2 = torch.nn.Conv1d(64, 32, 1)
        self.glm1_bn1_1 = nn.BatchNorm1d(32)
        self.glm1_bn1_2 = nn.BatchNorm1d(32)
        self.glm1_conv2 = torch.nn.Conv1d(32+32, 64, 1)
        self.glm1_bn2 = nn.BatchNorm1d(64)
        # MLP-2
        self.mlp2_conv1 = torch.nn.Conv1d(64, 64, 1)
        self.mlp2_bn1 = nn.BatchNorm1d(64)
        self.mlp2_conv2 = torch.nn.Conv1d(64, 128, 1)
        self.mlp2_bn2 = nn.BatchNorm1d(128)
        self.mlp2_conv3 = torch.nn.Conv1d(128, 512, 1)
        self.mlp2_bn3 = nn.BatchNorm1d(512)
        # GLM-2 (graph-contrained learning modulus)
        self.glm2_conv1_1 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_2 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_conv1_3 = torch.nn.Conv1d(512, 128, 1)
        self.glm2_bn1_1 = nn.BatchNorm1d(128)
        self.glm2_bn1_2 = nn.BatchNorm1d(128)
        self.glm2_bn1_3 = nn.BatchNorm1d(128)
        self.glm2_conv2 = torch.nn.Conv1d(128*3, 512, 1)
        self.glm2_bn2 = nn.BatchNorm1d(512)
        # MLP-3
        self.mlp3_conv1 = torch.nn.Conv1d(64+512+512+512, 256, 1)
        self.mlp3_conv2 = torch.nn.Conv1d(256, 256, 1)
        self.mlp3_bn1_1 = nn.BatchNorm1d(256)
        self.mlp3_bn1_2 = nn.BatchNorm1d(256)
        self.mlp3_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.mlp3_conv4 = torch.nn.Conv1d(128, 128, 1)
        self.mlp3_bn2_1 = nn.BatchNorm1d(128)
        self.mlp3_bn2_2 = nn.BatchNorm1d(128)
        # output
        self.output_conv = torch.nn.Conv1d(128, self.num_classes, 1)
        if self.with_dropout:
            self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x, a_s, a_l):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # MLP-1
        x = F.relu(self.mlp1_bn1(self.mlp1_conv1(x)))
        x = F.relu(self.mlp1_bn2(self.mlp1_conv2(x)))
        # FTM
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x_ftm = torch.bmm(x, trans_feat)
        # GLM-1
        sap = torch.bmm(a_s, x_ftm)
        sap = sap.transpose(2, 1)
        x_ftm = x_ftm.transpose(2, 1)
        x = F.relu(self.glm1_bn1_1(self.glm1_conv1_1(x_ftm)))
        glm_1_sap = F.relu(self.glm1_bn1_2(self.glm1_conv1_2(sap)))
        x = torch.cat([x, glm_1_sap], dim=1)
        x = F.relu(self.glm1_bn2(self.glm1_conv2(x)))
        # MLP-2
        x = F.relu(self.mlp2_bn1(self.mlp2_conv1(x)))
        x = F.relu(self.mlp2_bn2(self.mlp2_conv2(x)))
        x_mlp2 = F.relu(self.mlp2_bn3(self.mlp2_conv3(x)))
        if self.with_dropout:
            x_mlp2 = self.dropout(x_mlp2)
        # GLM-2
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = torch.bmm(a_s, x_mlp2)
        sap_2 = torch.bmm(a_l, x_mlp2)
        x_mlp2 = x_mlp2.transpose(2, 1)
        sap_1 = sap_1.transpose(2, 1)
        sap_2 = sap_2.transpose(2, 1)
        x = F.relu(self.glm2_bn1_1(self.glm2_conv1_1(x_mlp2)))
        glm_2_sap_1 = F.relu(self.glm2_bn1_2(self.glm2_conv1_2(sap_1)))
        glm_2_sap_2 = F.relu(self.glm2_bn1_3(self.glm2_conv1_3(sap_2)))
        x = torch.cat([x, glm_2_sap_1, glm_2_sap_2], dim=1)
        x_glm2 = F.relu(self.glm2_bn2(self.glm2_conv2(x)))
        # GMP
        x = torch.max(x_glm2, 2, keepdim=True)[0]
        # Upsample
        x = torch.nn.Upsample(n_pts)(x)
        # Dense fusion
        x = torch.cat([x, x_ftm, x_mlp2, x_glm2], dim=1)
        # MLP-3
        x = F.relu(self.mlp3_bn1_1(self.mlp3_conv1(x)))
        x = F.relu(self.mlp3_bn1_2(self.mlp3_conv2(x)))
        x = F.relu(self.mlp3_bn2_1(self.mlp3_conv3(x)))
        if self.with_dropout:
            x = self.dropout(x)
        x = F.relu(self.mlp3_bn2_2(self.mlp3_conv4(x)))
        # output
        x = self.output_conv(x)
        x = x.transpose(2,1).contiguous()
        x = torch.nn.Softmax(dim=-1)(x.view(-1, self.num_classes))
        x = x.view(batchsize, n_pts, self.num_classes)

        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeshSegNet().to(device)
    summary(model, [(15, 6000), (6000, 6000), (6000, 6000)])
