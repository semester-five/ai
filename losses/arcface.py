import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.mobilefacenet import l2_norm

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        # Tính sẵn các hằng số lượng giác để tối ưu tốc độ tính toán
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)     # Ngưỡng threshold để kiểm tra góc chết
        self.mm = self.sin_m * m # Lượng bù trừ khi rơi vào góc chết

    def forward(self, embedding, label):
        # 1. Chuẩn hóa L2 cho đặc trưng và trọng số
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        weight_norm = l2_norm(self.weight, axis=0)

        # 2. Tính cos(theta) thông qua tích vô hướng
        cosine = torch.mm(embedding_norm, weight_norm)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # 3. Tính cos(theta + m) bằng công thức: cos(A+B) = cosA.cosB - sinA.sinB
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m 

        # 4. Xử lý "Góc chết" (khi theta + m > pi)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 5. Chỉ áp dụng margin cho lớp đúng, giữ nguyên cos(theta) cho các lớp khác
        output = cosine * 1.0
        idx_ = torch.arange(0, embedding.size(0), dtype=torch.long)
        output[idx_, label.long()] = phi[idx_, label.long()]

        # 6. Scale
        output *= self.s

        return output