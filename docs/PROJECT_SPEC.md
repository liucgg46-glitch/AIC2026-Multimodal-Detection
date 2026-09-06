# AIC2026 Multimodal Detection Project

## Competition Task

AIC2026 算法挑战赛：
面向城市场景的视觉多模态目标检测。

输入模态：

1. visible：RGB 可见光图像
2. infrared：红外图像
3. depth：深度图像

任务：

根据三模态信息检测图像中的目标并输出 bounding boxes 和类别。

## Main Objective

项目第一目标：

在14天内完成可复现的初赛有效提交。

长期目标：

进入半决赛并冲击国奖。

## Development Strategy

按照以下顺序开发：

1. 数据检查
2. RGB-only baseline
3. IR-only baseline
4. Depth-only baseline
5. 简单多模态融合
6. RGB+IR+Depth Early Fusion
7. Feature-level Fusion
8. Adaptive / Reliability Fusion
9. 小目标优化
10. 最终比赛优化

## Base Framework

优先使用：

- Python
- PyTorch
- Ultralytics YOLO

禁止在没有明确理由的情况下切换整个训练框架。

## Development Principle

每次只实现一个明确功能。

每次算法修改都必须能够通过实验验证。

不要同时修改多个主要变量。

所有代码必须保证可复现。