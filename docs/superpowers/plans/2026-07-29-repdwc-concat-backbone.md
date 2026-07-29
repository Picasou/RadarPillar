# RepDWC concat backbone 重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 b5-b8/n4 的 `RepDWCNoneBackbone` 从「取首层直出」改成「RepBlock 多尺度 → ConvTranspose deblock → concat」，使其与 b1-b4 的 `BaseBEVBackbone` 仅在 block 类型上不同（公平对照）；新增 b9。

**Architecture:** 不动 `RepDWCBackbone`（继续返回多尺度 list），仅在 `RepDWCNoneBackbone` 内部新增 deblocks（ConvTranspose，与 `BaseBEVBackbone` 同构），把三层多尺度上采样到 160×160 后 concat。YAML 加 `UPSAMPLE_STRIDES`/`NUM_UPSAMPLE_FILTERS`，b9 按模板生成训练脚本。

**Tech Stack:** PyTorch, OpenPCDet, EasyDict YAML config, bash 训练脚本。

## Global Constraints

- 保持类名 `RepDWCNoneBackbone`、文件 `pcdet/models/backbones_2d/repdwc_none.py` 不变（用户决定不改名）。
- `RepDWCBackbone`（`rep_dwc.py`）零改动。
- deblock 用 `nn.ConvTranspose2d`，与 `BaseBEVBackbone` 完全同构（可学习上采样，公平对照）。
- 所有配置 `BATCH_SIZE_PER_GPU:16`。
- Python: `/home/admin/anaconda3/bin/python`，env=base，训练脚本需 `PYTHONPATH` 自动处理（脚本内已 conda activate）。
- GPU 3070Ti 8G；1-epoch 验证用 bs=16。
- 运行 conda env 探测顺序：脚本内 `find_conda_env` 优先 `angle`→`base`，本机 `angle` 不存在会落到 base（符合 ground-truth）。
- 用 Edit 精确匹配 old_string 改文件；CLAUDE.md 约定「新增函数需与用户沟通确认」——本重构不新增公开函数，仅改 `RepDWCNoneBackbone.__init__/forward` 内部实现。

## 已验证事实（实测，非臆测）

- `RepDWCBackbone` 输入 `(B,32,320,320)`、`LAYER_STRIDES=[2,2,2]` 时，三层 outs = `[(B,C,160,160),(B,C,80,80),(B,C,40,40)]`。
- ConvTranspose `stride=1/2/4`（kernel 同 stride）→ 三层全对齐到 `160×160`。
- `num_bev_features = sum(NUM_UPSAMPLE_FILTERS)` 与 `BaseBEVBackbone` 语义一致（`detector3d_template.py:100` 读此值喂 head）。

---

### Task 1: 重写 `RepDWCNoneBackbone` 为 deblock+concat

**Files:**
- Modify: `pcdet/models/backbones_2d/repdwc_none.py`（整个类重写）

**Interfaces:**
- Consumes: `RepDWCBackbone`（`rep_dwc.py`，返回 `list[Tensor]`，`self.num_outputs`）。`BaseBEVBackbone` 的 deblock 构造范式（`base_bev_backbone.py:47-69`）。
- Produces: `RepDWCNoneBackbone.__init__(model_cfg, input_channels=32)` 读 `OUT_CHANNELS/UPSAMPLE_STRIDES/NUM_UPSAMPLE_FILTERS`；`self.num_bev_features = sum(NUM_UPSAMPLE_FILTERS)`；`forward(data_dict)` 写 `data_dict['spatial_features_2d'] = cat(deblocks(outs))`。

- [ ] **Step 1: 整体替换 `repdwc_none.py` 内容**

用 Edit 把整个文件替换为：

```python
"""RPiN：RepBlock 多尺度 → ConvTranspose deblock → concat。

对齐 BaseBEVBackbone 的多尺度融合拓扑，使 RepDWC 实验组（b5-b9/n4）与
standard 组（b1-b4）仅在 block 类型（RepBlock vs 标准 3×3 Conv）上不同，
构成公平对照。RepDWCBackbone 零改动，本类在其 list 输出之上加 deblock+concat。
"""
import torch
import torch.nn as nn

from .rep_dwc import RepDWCBackbone


class RepDWCNoneBackbone(nn.Module):
    """RepBlock 多尺度 → ConvTranspose deblock → concat（对齐 BaseBEVBackbone）。"""

    def __init__(self, model_cfg, input_channels: int = 32):
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = RepDWCBackbone(model_cfg, input_channels)  # 返回 list[Tensor]

        out_channels = list(model_cfg.OUT_CHANNELS)
        upsample_strides = list(model_cfg.UPSAMPLE_STRIDES)
        num_upsample_filters = list(model_cfg.NUM_UPSAMPLE_FILTERS)
        assert len(num_upsample_filters) == self.backbone.num_outputs, \
            'NUM_UPSAMPLE_FILTERS 数量须等于 NUM_OUTPUTS'
        assert len(num_upsample_filters) == len(upsample_strides) == len(out_channels)

        self.deblocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(out_channels[i], num_upsample_filters[i],
                                   upsample_strides[i], stride=upsample_strides[i], bias=False),
                nn.BatchNorm2d(num_upsample_filters[i], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
            for i in range(len(num_upsample_filters))
        ])
        self.num_bev_features = int(sum(num_upsample_filters))

    def forward(self, data_dict):
        outs = self.backbone(data_dict)                      # [160×160, 80×80, 40×40]
        ups = [db(o) for db, o in zip(self.deblocks, outs)]  # 各上采样到 160×160
        data_dict['spatial_features_2d'] = torch.cat(ups, dim=1)
        return data_dict
```

- [ ] **Step 2: 改测试为 concat 断言**

Edit `tests/rpin/test_rpin_modules.py`，把 `test_repdwcnone_outs0_design` 整个函数替换为：

```python
def test_repdwcnone_concat_design():
    """n4: RepBlock 多尺度 → deblock → concat（替代旧的首层直出）。"""
    from pcdet.models.backbones_2d.repdwc_none import RepDWCNoneBackbone
    mcfg = _load_model_cfg('n4').BACKBONE_2D
    in_ch = int(_load_model_cfg('n4').MAP_TO_BEV.NUM_BEV_FEATURES)
    m = RepDWCNoneBackbone(mcfg, input_channels=in_ch)
    assert m.num_bev_features == 448                       # sum([64,128,256])
    bd = {'spatial_features': torch.randn(1, in_ch, 320, 320)}
    sf2d = m(bd)['spatial_features_2d']
    assert tuple(sf2d.shape[1:]) == (448, 160, 160)        # 3 层 concat
```

- [ ] **Step 3: 跑测试验证通过**

Run: `cd /home/admin/projects/RadarPillar && PYTHONPATH=tools /home/admin/anaconda3/bin/python -m pytest tests/rpin/test_rpin_modules.py::test_repdwcnone_concat_design -v`
Expected: PASS（1 passed）

- [ ] **Step 4: 提交**

```bash
git add pcdet/models/backbones_2d/repdwc_none.py tests/rpin/test_rpin_modules.py
git commit -m "feat(repdwc): 重构为 deblock+concat 以公平对照 standard"
```

---

### Task 2: 改 b5-b8、n4 的 YAML 加 deblock 配置

**Files:**
- Modify: `experiments/YAML/b5.yaml`, `b6.yaml`, `b7.yaml`, `b8.yaml`, `n4.yaml`

**Interfaces:**
- Consumes: Task 1 的 `RepDWCNoneBackbone`（需 `UPSAMPLE_STRIDES`/`NUM_UPSAMPLE_FILTERS`）。
- Produces: YAML 能被 detector 正确加载，`BACKBONE_2D` 含完整 deblock 配置。

每个文件的改法相同：在 `BACKBONE_2D` 块内，删 `NUM_OUTPUTS/INFERENCE_MODE/USE_SE/NUM_CONV_BRANCHES/USE_NORMCONV`（RepDWCBackbone 有默认值，留着也无害——但为干净保留它们），在 `OUT_CHANNELS` 后新增 `UPSAMPLE_STRIDES:[1,2,4]` + `NUM_UPSAMPLE_FILTERS:<同 OUT_CHANNELS>`。

> 决策：**保留** `NUM_OUTPUTS/INFERENCE_MODE/USE_SE/NUM_CONV_BRANCHES/USE_NORMCONV/USE_DWCONV` 不删（RepDWCBackbone 读取它们，删了会触发默认值，行为不变但减少显式控制）。仅**新增**两个键。

- [ ] **Step 1: 改 b5.yaml**

Edit `experiments/YAML/b5.yaml`，把：
```yaml
    OUT_CHANNELS:
    - 32
    - 32
    - 32
    NUM_OUTPUTS: 3
```
替换为：
```yaml
    OUT_CHANNELS:
    - 32
    - 32
    - 32
    UPSAMPLE_STRIDES:
    - 1
    - 2
    - 4
    NUM_UPSAMPLE_FILTERS:
    - 32
    - 32
    - 32
    NUM_OUTPUTS: 3
```

- [ ] **Step 2: 改 b6.yaml**

Edit `experiments/YAML/b6.yaml`，把：
```yaml
    OUT_CHANNELS:
    - 32
    - 64
    - 128
    NUM_OUTPUTS: 3
```
替换为：
```yaml
    OUT_CHANNELS:
    - 32
    - 64
    - 128
    UPSAMPLE_STRIDES:
    - 1
    - 2
    - 4
    NUM_UPSAMPLE_FILTERS:
    - 32
    - 64
    - 128
    NUM_OUTPUTS: 3
```

- [ ] **Step 3: 改 b7.yaml**

Edit `experiments/YAML/b7.yaml`，把：
```yaml
    OUT_CHANNELS:
    - 64
    - 128
    - 256
    NUM_OUTPUTS: 3
```
替换为：
```yaml
    OUT_CHANNELS:
    - 64
    - 128
    - 256
    UPSAMPLE_STRIDES:
    - 1
    - 2
    - 4
    NUM_UPSAMPLE_FILTERS:
    - 64
    - 128
    - 256
    NUM_OUTPUTS: 3
```

- [ ] **Step 4: 改 b8.yaml**

Edit `experiments/YAML/b8.yaml`，把：
```yaml
    OUT_CHANNELS:
    - 64
    - 64
    - 64
    NUM_OUTPUTS: 3
```
替换为：
```yaml
    OUT_CHANNELS:
    - 64
    - 64
    - 64
    UPSAMPLE_STRIDES:
    - 1
    - 2
    - 4
    NUM_UPSAMPLE_FILTERS:
    - 64
    - 64
    - 64
    NUM_OUTPUTS: 3
```

- [ ] **Step 5: 改 n4.yaml**

Edit `experiments/YAML/n4.yaml`，把：
```yaml
    OUT_CHANNELS:
    - 64
    - 128
    - 256
    NUM_OUTPUTS: 3
```
替换为：
```yaml
    OUT_CHANNELS:
    - 64
    - 128
    - 256
    UPSAMPLE_STRIDES:
    - 1
    - 2
    - 4
    NUM_UPSAMPLE_FILTERS:
    - 64
    - 128
    - 256
    NUM_OUTPUTS: 3
```

- [ ] **Step 6: 验证 5 个 YAML 均能加载且 num_bev_features 正确**

Run:
```bash
cd /home/admin/projects/RadarPillar && PYTHONPATH=tools /home/admin/anaconda3/bin/python -c "
from pcdet.config import cfg_from_yaml_file
expect = {'b5':96,'b6':224,'b7':448,'b8':192,'n4':448}
for tag,exp in expect.items():
    cfg = cfg_from_yaml_file(f'experiments/YAML/{tag}.yaml', None)
    b2d = cfg.MODEL.BACKBONE_2D
    nf = sum(b2d.NUM_UPSAMPLE_FILTERS)
    assert nf==exp, f'{tag}: {nf}!={exp}'
    print(f'{tag}: num_bev_features={nf} OK')
"
```
Expected: 5 行 `OK`（b5=96 b6=224 b7=448 b8=192 n4=448）

- [ ] **Step 7: 提交**

```bash
git add experiments/YAML/b5.yaml experiments/YAML/b6.yaml experiments/YAML/b7.yaml experiments/YAML/b8.yaml experiments/YAML/n4.yaml
git commit -m "feat(repdwc): b5-b8/n4 YAML 加 deblock 配置(对齐 standard 多尺度融合)"
```

---

### Task 3: 新增 b9 配置 + 训练脚本

**Files:**
- Create: `experiments/YAML/b9.yaml`
- Create: `experiments/SH/train_rpillar_b9.sh`
- Reference: `experiments/YAML/b8.yaml`（b9 基于 b8 结构，OUT 改 [16,16,16]），`.claude/skills/model-train/templates/train.sh.template`

**Interfaces:**
- Consumes: Task 1 的 `RepDWCNoneBackbone`；b8.yaml 作为 YAML 模板；train.sh.template 作为 sh 模板。
- Produces: `experiments/YAML/b9.yaml`（RepDWC concat，OUT/UPSAMPLE=[16,16,16]，concat=48），`train_rpillar_b9.sh`（可执行训练脚本）。

- [ ] **Step 1: 创建 b9.yaml**

基于 b8.yaml 全文，仅改 BACKBONE_2D 的 OUT_CHANNELS / NUM_UPSAMPLE_FILTERS 为 [16,16,16]，EXTRA_TAG 文档行改为 b9。最稳做法：先复制 b8.yaml，再 Edit 两个值块 + 顶部注释。

Run:
```bash
cd /home/admin/projects/RadarPillar && cp experiments/YAML/b8.yaml experiments/YAML/b9.yaml
```

然后 Edit `experiments/YAML/b9.yaml`：
- 把 `OUT_CHANNELS:` 块 `- 64 / - 64 / - 64` 改为 `- 16 / - 16 / - 16`
- 把 `NUM_UPSAMPLE_FILTERS:` 块 `- 64 / - 64 / - 64` 改为 `- 16 / - 16 / - 16`

（b8 的这两个块当前都是 `[64,64,64]`，逐一 Edit）

- [ ] **Step 2: 创建 train_rpillar_b9.sh**

基于模板生成。b5.sh 与模板同构，复制 b5.sh 改 3 处：

Run:
```bash
cd /home/admin/projects/RadarPillar && cp experiments/SH/train_rpillar_b5.sh experiments/SH/train_rpillar_b9.sh
```

Edit `experiments/SH/train_rpillar_b9.sh`：
- 注释行 `train_rpillar_b5.sh` → `train_rpillar_b9.sh`
- `CFG_FILE:=experiments/YAML/b5.yaml` → `experiments/YAML/b9.yaml`
- `EXTRA_TAG:=rp_base_0716` → `b9`

- [ ] **Step 3: 验证 b9 配置加载正确**

Run:
```bash
cd /home/admin/projects/RadarPillar && PYTHONPATH=tools /home/admin/anaconda3/bin/python -c "
from pcdet.config import cfg_from_yaml_file
cfg = cfg_from_yaml_file('experiments/YAML/b9.yaml', None)
b2d = cfg.MODEL.BACKBONE_2D
assert sum(b2d.NUM_UPSAMPLE_FILTERS)==48, sum(b2d.NUM_UPSAMPLE_FILTERS)
assert list(b2d.OUT_CHANNELS)==[16,16,16]
print('b9: num_bev_features=48 OK')
"
```
Expected: `b9: num_bev_features=48 OK`

- [ ] **Step 4: 赋予执行权限并提交**

```bash
cd /home/admin/projects/RadarPillar && chmod +x experiments/SH/train_rpillar_b9.sh
git add experiments/YAML/b9.yaml experiments/SH/train_rpillar_b9.sh
git commit -m "feat(repdwc): 新增 b9 小容量档[16,16,16] + 训练脚本"
```

---

### Task 4: 1-epoch 训练验证（b5-b9, n4）

**Files:**
- 无改动，纯验证。消耗 `experiments/SH/train_rpillar_*.sh`。

**Interfaces:**
- Consumes: Task 1-3 的全部产物。

- [ ] **Step 1: 跑 b5 1-epoch foreground**

Run: `cd /home/admin/projects/RadarPillar && EPOCHS=1 RUN_MODE=foreground bash experiments/SH/train_rpillar_b5.sh`
Expected: 正常起训，无 shape/断言/配置错误，1 epoch 完成。关注日志末尾无 `Traceback`/`RuntimeError`/`AssertionError`，且 `spatial_features_2d` shape 正确流入 head。

- [ ] **Step 2-6: 依次跑 b6/b7/b8/b9/n4 1-epoch foreground**

对每个 tag：
Run: `EPOCHS=1 RUN_MODE=foreground bash experiments/SH/train_rpillar_<tag>.sh`
Expected: 同上，无报错。

> 注意：bs=16 在 3070Ti 8G 上，b3/b4/b7/b8/b9 大容量档可能 OOM。若某档 OOM，记录现象，**不改 bs**（spec 锁定全 16），报告用户决策。其余档应正常。

- [ ] **Step 7: 汇总验证结果**

确认 6 个配置（b5/b6/b7/b8/b9/n4）均 1-epoch 无报错（或明确记录 OOM 的档位）。如实报告：通过数 / OOM 数，附各日志关键行。

---

## Self-Review

**1. Spec 覆盖：**
- repdwc_none.py 改 concat → Task 1 ✓
- b5-b8+n4 YAML 加 deblock → Task 2 ✓
- 新增 b9 → Task 3 ✓
- 训练脚本 b9 → Task 3 Step 2 ✓
- 测试断言改写 → Task 1 Step 2 ✓
- 1-epoch 验证 → Task 4 ✓
- bs 全 16 → Task 2/3/4 明确 ✓

**2. Placeholder：** 无 TBD/TODO；每个 code step 有完整代码。

**3. 类型一致性：** `RepDWCNoneBackbone.__init__(model_cfg, input_channels=32)` / `num_bev_features` / `forward` 签名跨任务一致；YAML 键名 `UPSAMPLE_STRIDES`/`NUM_UPSAMPLE_FILTERS` 与 Task1 assert 一致。
