# 编码孔径目录

请将正式编码孔径及其元数据放在本目录：

```text
data/mask/mask.npy
data/mask/mask.meta.json
```

`mask.npy` 的协议要求：

- 形状：`[256, 256]`
- 类型：`float32`
- 数值：仅允许二值 `0` 和 `1`
- 所有元素必须有限，不能包含 `NaN` 或 `Inf`

`mask.meta.json` 至少需要记录：

- `schema_version`
- `mask_id`
- `role`（正式孔径使用 `formal`）
- `shape`
- `dtype`
- `binary`
- `open_fraction`
- `sha256`
- 孔径来源或生成方式

当前项目的正式仿真基线由原项目连续透过率 Mask 派生：保留源空间
排序，将透过率最高的 50% 像素设为 1，其余设为 0。来源副本保存在
`data/mask/source/legacy_project_mask.npy`。该派生 Mask 可用于正式仿真
基线，但未经真实硬件标定，不能表述为实测物理孔径。

开发随机孔径使用 `dev_*.npy` 命名，并标记为
`development_smoke_only`，不能用于生成正式训练数据。
