# Dataset Specification

## Current Sample Dataset

数据根目录包含：

data/sample_raw/

├── visible/  
├── infrared/  
├── depth/  
└── labels/  

## Modalities

### Visible

RGB 可见光图像。

### Infrared

红外图像。

具体通道数、dtype 和尺寸由数据检查程序确认。

### Depth

深度图。

读取时必须保持原始数据类型。

禁止默认按照普通 8-bit RGB 图像处理。

必须使用类似：

cv2.imread(path, cv2.IMREAD_UNCHANGED)

检查：

- dtype
- min
- max
- invalid value
- zero ratio

### Labels

当前标签格式必须通过实际样例确认。

在没有确认标签格式前：

禁止自动转换标签。

## Alignment Requirement

visible、infrared、depth、labels 应通过文件 stem 一一对应。

例如：

visible/000001.xxx   
infrared/000001.xxx   
depth/000001.xxx  
labels/000001.txt  

必须检查：

- 文件是否缺失
- 图像空间尺寸是否一致
- 三模态是否空间对齐