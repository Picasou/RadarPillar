# Vitis AI 官方证据快照（2026-09-15）

部署链路环境审计（2026-09-15，内容已并入 [DEPLOY_GUIDE.md](../../DEPLOY_GUIDE.md)）联网核实时抓取的官方文档原文，防链接失效/改版。上游：https://github.com/Xilinx/Vitis-AI 与 https://xilinx.github.io/Vitis-AI/

- `version_compatibility.rst` / `docker_versions.rst`：版本兼容矩阵（VAI↔IP↔Vivado/PetaLinux↔kernel）与镜像 tag
- `workflow-system-integration.rst`（3.5 版）/ `wsi_v5.rst`（5.0 版同文）：MPSoC 系统集成工作流（Vivado/Vitis 两流程、PetaLinux 组件、CMA、vart vivado recipe 坑）
- `workflow-model-deployment.rst`：官方三步部署法（自定义板需 port platform）
- `README_DPUCZDX8G.md` / `README_DPUCZ_Vivado.md` / `README_DPUCZ_Vivado_sw.md` / `README_DPUCZ_Vitis.md` / `tutorial_vivado_trd.md`：DPU TRD（block design 集成、arch.json 产出位置、时钟/中断/DDR、PetaLinux 开关、DPU 参数变更须重编模型）
- `board_setup_mpsoc_v30.rst`：v3.0 预编译板卡镜像覆盖范围（ZCU102/104/KV260）
- `quick_start_for_embedded.md`：VART 交叉编译 sysroot/SDK 步骤（sdk-*.sh + vitis_ai sysroot 增量包）
- `custom_op_readme.md` / `pp_main.cpp`：CPU custom operator（官方 PointPillars PPScatter 先例）
- `vai_repo/board_setup/README.md`：Vitis-AI 仓库 sparse checkout 残留（仅此一文件，其余引用见审计文档参考来源）
