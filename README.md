# CASSI Core20 正式基线

本目录按《项目搭建工作流》实现从 84 波段 HSI 到 20 通道植被指数的四阶段流程：

1. HSI 有效性检查与 `reviewed_v3_core20` 标签生成；
2. CASSI 光学前向、物理 Mask 派生和测量样本生成；
3. `(measurement, model_mask) -> 20 通道指数` 的 Restormer 训练；
4. Patch 按原坐标拼接，并在原始指数空间执行场景宏平均评估。
5. 在同一 Core20 协议下执行 Transformer、Mamba、公式引导、简单直接预测和重建后计算的论文对照。

根目录配置已切换到服务器 252 景正式基线；三景配置仍保存在
`configs/local_smoke.json`，其结果只用于行为门禁，不是正式泛化结果。

## 环境

```bash
cd /home/yanzhi/project/CASSI/cassi
source /home/yanzhi/miniconda3/etc/profile.d/conda.sh
conda activate deep-learning
```

服务器使用：

```bash
cd /home/user/programs/cassi-core20
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate cassi
```

## 测试

```bash
pytest -q
```

## 主要入口

```bash
# 第一阶段：核验已生成的 HSI 有效域与标签
python code/hsi_valid_mask.py --existing verify
python code/lable.py --existing verify

# 第二阶段：生成并核验正式 252 景测量集
python code/measurement_dataset.py generate-formal \
  --split-manifest data/manifests/formal_252_split_seed42_202_15_35.json
python code/measurement_dataset.py verify-formal

# 本地三景核验
python code/measurement_dataset.py verify-minimum

# 第三阶段：配置驱动训练；实验目录存在时拒绝覆盖
python code/fit_normalization.py --config config.json
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python code/smoke_cuda.py --config config.json
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python code/train.py --config config.json

# 从每轮结束时保存的完整状态继续
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python code/train.py --config config.json --resume

# 本地三景训练使用保留配置
python code/train.py --config configs/local_smoke.json

# 第四阶段：本地验证诊断
python code/evaluate.py --config config.json --split validation
```

各入口的完整参数以 `--help` 为准。正式训练唯一可调配置源是项目根目录的
`config.json`，相对路径均按项目根目录解析。逐字段中文说明见
`config.example.jsonc`；它包含注释，因此只供阅读，不能直接作为训练配置。

## 当前协议边界

- `measurement_scale=0.9` 只在 CASSI 前向中应用一次，是历史仿真约定，不是硬件透过率标定。
- 正式仿真 Mask 是从原项目连续值 Mask 按空间排序派生的固定 50% 二值孔径；源文件和派生元数据均保留。
- `index_valid_mask` 只进入损失和评估，不进入模型输入。
- 模型输出位于标准化目标空间；评估前按训练集统计量反变换到原始指数空间。
- `configs/local_smoke.json` 的验证只覆盖 `hsi_0003` 的两个 256×256 Patch；正式配置改用冻结划分中的 15 个验证场景。完整 HWC 预测在未覆盖区域保存为 NaN，且这些区域不会进入指标。
- 正式测试必须先迁移并核验 `formal_252_split_seed42_202_15_35` 的 202/15/35 场景划分，冻结协议后再运行一次 35 景测试。
- `evaluate.py --split test` 会强制检查正式划分 ID、202/15/35 数量及测量清单中的划分身份，避免把本地结果误标为正式测试。
- 训练入口通过项目内的 `SceneMacroRunnerV3(RunnerV3)` 只改写验证汇总：先按坐标执行重叠均值融合，再逐指数、逐场景等权计算 `macro_zMAE`；外部学习仓库源码保持不变。

## 第五阶段计划

- 必做直接预测模型：`RestormerCore20`、`MST-MambaCore20`、`IFGNetCore20` 和 `UNetCore20`。
- 必做传统基线：CASSI 测量重建 84 波段 HSI 后，再用同一 `reviewed_v3_core20` 公式计算 20 指数。
- `DHM` 只在 MST-Mamba 值得继续时作全局/局部 Mamba 消融；`WPO3D` 只在完成与 CASSI 成像方程的数学对应后纳入。
- 首轮固定一个种子使用训练集/验证集筛选；最终保留方法至少运行 3 个随机种子，模型和超参数冻结后才进行一次正式测试。
- 旧仓库 32 通道权重和重复归一化测试数值只作历史记录，不与 Core20 结果混用。

## 证据

阶段验收记录位于 `artifacts/acceptance/`；训练产物和评估结果位于
`outputs/<experiment_id>/`。`outputs/` 和大型 NPY 均被 Git 忽略，JSON 元数据与验收记录保留。
正式训练的控制台输出同时写入 `outputs/formal_restormer_core20_seed42_v1/train.log`；
每轮结束会原子保存 `history.json` 和 `last_training_state.pt`。只有训练正常结束或满足提前停止条件后，
才会生成 `formal_training_report.json`；后台进程仍在运行不等于训练已经完成。
