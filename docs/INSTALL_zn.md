# 安装指南

### 环境要求

所有代码均在以下环境中测试通过：

* Linux（在 Ubuntu 14.04 / 16.04 上测试）
* Python 3.6+
* PyTorch 1.1 或更高（在 PyTorch 1.1、1.3、1.5 上测试）
* CUDA 9.0 或更高（PyTorch 1.3+ 需要 CUDA 9.2+）
* [`spconv v1.0（commit 8da6f96）`](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) 或 [`spconv v1.2`](https://github.com/traveller59/spconv)

### 安装 `pcdet v0.3`

注意：即使已经安装过旧版本，请务必重新运行 `python setup.py develop` 来安装 `pcdet v0.3`。

a. 克隆本仓库：

```shell
git clone https://github.com/open-mmlab/OpenPCDet.git
```

b. 按如下步骤安装依赖库：

* 安装 Python 依赖库：

```
pip install -r requirements.txt
```

* 安装 SparseConv 库，我们使用 [`[spconv]`](https://github.com/traveller59/spconv) 的实现：
    * 如果你使用的是 PyTorch 1.1，请确认安装的是 [`commit 8da6f96`](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) 对应的 `spconv v1.0`，而不是最新版。
    * 如果你使用的是 PyTorch 1.3+，则需要安装 `spconv v1.2`。正如 [`spconv`](https://github.com/traveller59/spconv) 作者所述，如果你使用 PyTorch 1.4+，需要使用其官方 Docker 镜像。

c. 通过运行以下命令安装 `pcdet` 库：

```shell
python setup.py develop
```
