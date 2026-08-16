# 模型目录说明

本目录存放 OpenVINO IR 格式的装甲板检测模型，

请在此处放置或软链以下两个文件（OpenVINO IR 由 `.xml` + `.bin` 成对组成）：

- `armor.xml`
- `armor.bin`

`config/app/auto_aim/detector.toml` 中的 `[inference] model_path` 默认指向 `data/models/armor.xml`。

模型需满足以下输入/输出协议：

- 输入：`[1, 3, H, W]`（NCHW），静态 H/W。
- 输出：`[1, N, 22]`，`f32`，其中每行 22 个字段对应 YOLOv5 装甲板关键点检测协议。
