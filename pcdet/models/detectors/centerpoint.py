"""CenterPoint-style detector for the RadarNeXt port (audit B: NEW detector).

Why a new detector?
-------------------
OpenPCDet's PointPillar-style detectors (PointPillar/SECONDNet/...) share
``Detector3DTemplate`` whose forward expects the dense head to expose the
anchor-based ``box_cls_labels``/``reg_targets`` contract and whose eval path
calls ``Detector3DTemplate.post_processing`` — an anchor/box-NMS routine that
operates on ``batch_dict['batch_cls_preds']`` / ``batch_box_preds``.

The RadarNeXt head is anchor-free CenterPoint-style: it owns its own heatmap
NMS + rectifier scoring in ``post_processing`` and produces predictions during
its own forward (``data_dict['pred_dicts']``). Wiring it through the template's
anchor post-processing would corrupt it.

So this ``CenterPoint`` detector is a thin Detector3DTemplate subclass that:
  * builds the same module_topology (vfe, map_to_bev, backbone_2d, dense_head),
  * in training: calls ``self.dense_head.get_loss()``,
  * in eval: uses the head-produced ``data_dict['pred_dicts']`` directly,
    bypassing ``Detector3DTemplate.post_processing``.

The dense head (``RadarNeXtCenterHead``) is registered separately and selected
via ``DENSE_HEAD.NAME`` in the YAML.
"""

import torch
from .detector3d_template import Detector3DTemplate


class CenterPoint(Detector3DTemplate):
    """Anchor-free CenterPoint-style detector (RadarNeXt port)."""

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.init_weights()

    def init_weights(self):
        """Conv2d Kaiming + BN weight uniform_ across the whole detector
        (audit #10). Mirrors RadarNeXt ``init_weights``; the dense head already
        initializes itself, but we re-apply the scheme network-wide for any
        backbone/neck BN tensors."""
        # 严格对齐论文(原工程 radarnext.py:77-80): detector 顶层只对 BN weight 做 uniform_, **不动 Conv**
        # 避免覆盖 backbone/neck (RepDWC/DCNv3/MDFEN) 自身的 Kaiming init, 也避免覆盖
        # dense_head 填的 init_bias=-2.19 (SepHead 末层 Conv2d.bias)。原工程 mmengine 默认 init_cfg
        # 也只动 BN, Conv bias 由子模块自己负责(这里是关键 paper-spec 偏离)。
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                if m.weight is not None:
                    torch.nn.init.uniform_(m.weight)

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            # The dense head has already produced final predictions in
            # batch_dict['pred_dicts']; no anchor post-processing is run.
            pred_dicts = batch_dict.get('pred_dicts', [])
            recall_dicts = {}
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {'loss_rpn': loss_rpn.item(), **tb_dict}
        loss = loss_rpn
        return loss, tb_dict, disp_dict
