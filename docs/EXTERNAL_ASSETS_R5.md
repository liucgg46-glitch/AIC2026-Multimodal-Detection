# Round 5 外部源码与权重身份

本文件记录的是实际下载并计算得到的字节哈希。二进制文件受 `.gitignore` 排除，不进入 Git；训练入口和准备脚本会在运行时复核。

| 资产 | 官方来源 | 固定身份 |
|---|---|---|
| YOLO11x COCO 权重 | Ultralytics `v8.3.0/yolo11x.pt` | SHA256 `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24` |
| DEIMv2 源码 | `Intellindust-AI-Lab/DEIMv2` | commit `1d2ca42171570c713e78fc6a766ec5104b7f4724` |
| DEIMv2-S DINOv3 COCO `.pth` | 官方 Model Zoo Google Drive 文件 `1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL` | SHA256 `9491ab33b68ecfc0e34043abb3009599ab1e892fb953a1faad12ef4fca5a35c4` |
| ViT-Tiny distilled from DINOv3-S | 官方 Google Drive 文件 `1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs` | SHA256 `2053b865f4e2673fba3f95f7e7e54ad5ee18143885e3ad27eaabb5b3b9919738` |
| Hugging Face DEIMv2-S 参考发布 | `Intellindust/DEIMv2_DINOv3_S_COCO` revision `cf0540f3f319bb8ecbe132624358a35a7beb86d5` | `model.safetensors` LFS SHA256 `fe545b150766b8761a696f7b1d92ea138fafe8e02a7398823e0c62d2a5867956` |

正式微调使用 Model Zoo 的 `.pth`，因为上游 `train.py -t` 直接消费训练 checkpoint；Hugging Face safetensors 只作为同一官方模型发布的交叉参考，不混入正式运行。两个 Google Drive 文件均已用 `torch.load` 验证：检测器文件含 `model` state，骨干文件为 ViT 参数 `OrderedDict`。
