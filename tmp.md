# Aries 板新系统部署 · 全案记录（2026-10-12 ~ 10-15）

## 〇、当前状态（一行版）

**软件侧已通**（6.6 内核/驱动/SSH/密码/自动配网全部就绪），**剩余阻塞 = 网卡物理数据路**：a1 DTB 配 EMIO 绕道 vs 板上电线实际接 MIO 直连——需 FPGA 侧改 ENET3 回 MIO 重导 XSA，或找到正确的 PHY 复位 GPIO 编号使 MDIO 能发现真 PHY。

## 一、编译与制卡链路（命令 → 产物）

```bash
# 编译五件（BOOT.BIN/boot.scr/image.ub/rootfs.tar.gz/sdk.sh）
PLNX_PROJ=~/axu4ev_dpu_2/petalinux bash deployment/board/petalinux/build_petalinux.sh \
    deployment/cfg/a1_2024.1/Aries_MAINBOARD.xsa

# 打包 store 部署包（含 md5 指纹交接单）
python deployment/pack_store.py <tag> --board

# 制卡: parted(p1 FAT32 1G + p2 ext4) → mkfs → cp fat32/* → tar rootfs → 三方 md5 对拍
```

产物来源：XSA（FPGA Vivado 导出）+ PetaLinux 2024.1 官方 + 本工程配置 + 三驱动（aries-hwa/aries-mem/ax-spi-drv）

## 二、关键实验矩阵

| 组件\实验 | E2/E4 | E8-E12 | **E17** | **final1-7** | **final8-10** | **final11(待)** |
|---|---|---|---|---|---|---|
| BOOT.BIN | 老系统 | 新(5EV位流) | 老系统 | 老系统 | 老系统 | 老系统 |
| 内核 | 5.4 | 6.6 | **官方6.6** | 官方重组6.6 | 官方重组6.6 | 官方重组6.6 |
| DTB | 老系统 | a1 | 官方 | a1+mii/gmii+fl | a1 / 老系统 | **a1+mii无fl** |
| 打包 | 厂商FIT | 官方 | **mkimage** | mkimage | mkimage | mkimage |
| 结果 | **活** | 死(dtc坑) | **✓到登录** | 起系统/网不通 | 起系统/网不通/老DTB可能内核死 | 待验 |

## 三、推理链

1. **E17 翻案**：官方 mkimage FIT + 老系统 BOOT = 6.6 完整启动 → 板/内核/XSA 全无辜
2. **15 连死真凶**：dtc 手拼 FIT（老 u-boot 把镜像头当设备树递给内核）→ 已全面禁用
3. **串口现形**：控制台与雷达数据共用一根线；FT232 交叉接法可收发
4. **网卡 MAC 层通**：mii+fixed-link → Link Up 100M/Full（但 fixed-link 是虚拟 PHY，不查真硬件）
5. **物理数据路不通**：ARP 零应答、MDIO 无设备 → MAC 走 EMIO 绕道而电线在 MIO 直连
6. **PHY 复位 GPIO**：老系统用 402/403；6.6 内核 GPIO base=580（zynqmp_gpio 174 脚），编号映射待确认
7. **同事建议（有效但需适配）**：用老系统一样的内容的 DTB（ENET 走 MIO/mii）+ user_setup.sh 操作 PHY 复位——但老 DTB 在 6.6 下可能不兼容（final10 串口只见雷达数据无内核输出）

## 四、已修复并固化的构建链病灶

① 位流混入陈旧件 → XSA 解包直贯+载荷门禁；② pack_store 默认工程指向死工程 → 自动探测；③ CFG_SHA 相对路径失效 → 绝对路径；④ IP 手改 tar → bbappend 源头；⑤ 管道尾吞退出码 → REAL_EXIT 落日志；⑥ spidev 内建 → 模块；⑦ FIT 重组一律 mkimage 禁 dtc；⑧ 内存盘嵌入前剥 64 字节 u-boot 头

## 五、网卡问题的三条候选路径

| 路径 | 做法 | 状态 |
|---|---|---|
| A: FPGA 改 XSA | ENET3 从 EMIO 改回 MIO 64-77 重导 XSA | 最正确但需 FPGA 同事 |
| B: 找到正确 GPIO | 在 6.6 内核扫描 GPIO 编号释放 PHY 复位 | GPIO 658/659 已试无效；老系统 402/403 的映射未确定 |
| C: mii 无 fixed-link | 内核在 MDIO 上找真 PHY（老 FSBL 可能已释放 PHY） | final11 待上板 |

## 六、产物与材料

- **最新可用镜像**：`.tmp/aries_rescue/final11.img`（mii 无 fixed-link + 内存盘含密码/SSH/配网）
- 回滚件：`deployment/backup_aries_2020.1/` + `.tmp/aries_rescue/p1old/`
- store 包基座：`deployment/store/*strict6/`
- 串口诊断：FT232 交叉接"串口"连接器，115200，COM4

## 七、FPGA 侧一句话任务

> Vivado 打开 a1，把 PS 的 ENET3 从 EMIO 改回 **MIO 64-75 + MDIO 76-77**（对齐老系统已验证布线），勾 Include bitstream 重导 XSA。
