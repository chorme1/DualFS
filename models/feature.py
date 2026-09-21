import torch
import torch.nn as nn
import torch.nn.functional as F


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        att = self.relu(self.w_g(g) + self.w_x(x))
        alpha = self.psi(att)
        return x * alpha


class DomainGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate_s = nn.Linear(channels, channels)
        self.gate_t = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat, domain: str):
        pooled = self.pool(feat).view(feat.size(0), -1)
        if domain == "source":
            gate = self.sigmoid(self.gate_s(pooled))
        else:
            gate = self.sigmoid(self.gate_t(pooled))
        return gate.view(gate.size(0), gate.size(1), 1, 1)


class FeatureNetwork(nn.Module):
    def __init__(self, feature_dim, src_input_dim, tar_input_dim, n_dim, class_num, e1_channels=64):
        super().__init__()
        base_ch = max(16, n_dim)

        self.target_mapping = Mapping(tar_input_dim, base_ch)
        self.source_mapping = Mapping(src_input_dim, base_ch)

        self.enc1 = ConvBlock(base_ch, e1_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(e1_channels, e1_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(e1_channels * 2, e1_channels * 4)

        self.up2 = nn.ConvTranspose2d(e1_channels * 4, e1_channels * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(e1_channels * 2, e1_channels * 2, e1_channels)
        self.dec2 = ConvBlock(e1_channels * 4, e1_channels * 2)
        self.up1 = nn.ConvTranspose2d(e1_channels * 2, e1_channels, kernel_size=2, stride=2)
        self.att1 = AttentionGate(e1_channels, e1_channels, max(8, e1_channels // 2))
        self.dec1 = ConvBlock(e1_channels * 2, e1_channels)

        self.fusion_seg_head = nn.Conv2d(e1_channels, class_num, kernel_size=1)

        self.gate = DomainGate(e1_channels)
        self.sep_head = nn.Conv2d(e1_channels, class_num, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_proj = nn.Linear(e1_channels * 4, feature_dim)

    def _encode(self, x, domain):
        if domain == "target":
            x = self.target_mapping(x)
        else:
            x = self.source_mapping(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        return e1, e2, b

    def _decode(self, e1, e2, b):
        d2 = self.up2(b)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))
        return d1

    def forward(self, x, domain="source"):
        e1, e2, b = self._encode(x, domain=domain)
        fusion_feat = self._decode(e1, e2, b)
        fusion_logits = self.fusion_seg_head(fusion_feat)

        gate = self.gate(e1, domain=domain)
        sep_feat = e1 * gate
        sep_logits = self.sep_head(sep_feat)

        pooled = self.global_pool(b).view(b.size(0), -1)
        e2_feat = self.feature_proj(pooled)

        return {
            "e1_feat": e1,
            "e2_feat": e2_feat,
            "gate": gate,
            "sep_logits": sep_logits,
            "fusion_logits": fusion_logits,
        }

    def forward_fusion(self, x, domain="source"):
        e1, e2, b = self._encode(x, domain=domain)
        fusion_feat = self._decode(e1, e2, b)
        fusion_logits = self.fusion_seg_head(fusion_feat)
        pooled = self.global_pool(b).view(b.size(0), -1)
        e2_feat = self.feature_proj(pooled)
        return e2_feat, fusion_logits
