# ComfyUI Mask Expansion 自定义节点

这是一个ComfyUI自定义节点,用于将严格遮罩转换为区域遮罩,支持多种扩展方式。

## 功能特性

### 1. Mask Expansion (严格遮罩到区域遮罩)
将输入的masks从严格遮罩转换为区域遮罩,支持多种扩展模式。

**输入参数:**
- `masks`: 输入的遮罩(MASK类型)
- `mode`: 扩展模式(必需)
  - `fixed_padding`: 固定padding扩展
  - `ratio_expand`: 按比例扩展
  - `morphology`: 形态学膨胀
  - `gaussian_blur`: 高斯模糊扩展

**可选参数:**
- `padding`: 固定padding值(像素) - 0-100, 默认10
- `expand_ratio`: 扩展比例 - 0.0-2.0, 默认0.2
- `kernel_size`: 形态学操作的核大小(奇数) - 1-31, 默认5
- `blur_sigma`: 高斯模糊的标准差 - 0.5-20.0, 默认2.0
- `blur_threshold`: 模糊后的二值化阈值 - 0.0-1.0, 默认0.3

**输出:**
- `expanded_masks`: 扩展后的masks

### 2. Mask Feathering (边缘羽化)
对mask边缘进行羽化处理,使mask边界更加柔和自然。

**输入参数:**
- `masks`: 输入的遮罩(MASK类型)
- `feather_size`: 羽化大小(像素) - 0-50, 默认5

**输出:**
- `feathered_masks`: 羽化后的masks

### 3. Mask Invert (反转遮罩)
反转mask,将黑色区域变成白色,白色区域变成黑色。

**输入参数:**
- `masks`: 输入的遮罩(MASK类型)

**输出:**
- `inverted_masks`: 反转后的masks

## 安装方法

1. 将 `MaskExpansionNode` 文件夹复制到你的ComfyUI自定义节点目录:
   ```
   ComfyUI/custom_nodes/MaskExpansionNode/
   ```

2. 安装依赖包:
   ```bash
   pip install scipy
   ```

3. 重启ComfyUI,节点将自动加载。

## 使用示例

### 基本用法流程

```
Mask Input → Mask Expansion → Feathering (可选) → 后续处理
```

### 场景1: 固定Padding扩展
当需要为mask添加固定的扩展区域时:
- 设置 `mode` 为 `fixed_padding`
- 调整 `padding` 参数(例如10像素)
- 适用于需要精确控制扩展量的场景

### 场景2: 按比例扩展
当需要根据mask大小动态调整扩展量时:
- 设置 `mode` 为 `ratio_expand`
- 调整 `expand_ratio` 参数(例如0.2表示扩展20%)
- 适用于不同大小的mask需要统一扩展比例的场景

### 场景3: 形态学膨胀
当需要保持mask形状并进行平滑扩展时:
- 设置 `mode` 为 `morphology`
- 调整 `kernel_size` 参数(较大的值会产生更大的扩展)
- 适用于需要自然形状扩展的场景

### 场景4: 高斯模糊扩展
当需要柔和的边界扩展时:
- 设置 `mode` 为 `gaussian_blur`
- 调整 `blur_sigma` 控制模糊程度
- 调整 `blur_threshold` 控制二值化阈值
- 适用于需要平滑过渡的场景

### 场景5: 边缘羽化
当需要让mask边缘更加柔和时:
- 使用 `Mask Feathering` 节点
- 设置 `feather_size` 控制羽化范围
- 适用于需要自然边缘的场景,比如人像抠图

## 工作原理

### Mask Expansion 节点

1. **Fixed Padding 模式**:
   - 检测mask的边界
   - 在边界外围添加固定像素的扩展
   - 适用于需要精确控制扩展量的场景

2. **Ratio Expand 模式**:
   - 检测mask的边界和尺寸
   - 根据mask尺寸计算扩展量
   - 适用于不同大小的mask需要统一扩展比例的场景

3. **Morphology 模式**:
   - 使用形态学膨胀操作
   - 通过结构元素扩展mask
   - 适用于需要保持形状的自然扩展

4. **Gaussian Blur 模式**:
   - 对mask进行高斯模糊
   - 通过阈值二值化生成扩展区域
   - 适用于需要柔和边界的场景

### Mask Feathering 节点

1. 计算mask到边界的距离
2. 在边缘区域创建平滑的过渡
3. 使用距离变换生成羽化权重

## 技术细节

- 支持批量处理多个mask
- 自动处理torch.Tensor和numpy格式
- 边界检查确保扩展不超出图像范围
- 使用float32类型保持精度

## 常见问题

**Q: 不同的扩展模式有什么区别?**
A: 
- Fixed Padding: 扩展量固定,适合需要精确控制的场景
- Ratio Expand: 根据mask大小按比例扩展,适合处理不同大小的对象
- Morphology: 保持mask形状的自然扩展,适合需要平滑边界的场景
- Gaussian Blur: 柔和的边界扩展,适合需要平滑过渡的场景

**Q: 如何选择合适的kernel_size?**
A: kernel_size越大,扩展效果越明显。建议从小的值(如3或5)开始尝试,逐步增加直到达到满意的效果。

**Q: 高斯模糊扩展中threshold参数的作用是什么?**
A: threshold控制模糊后的二值化阈值。较低的值会产生更大的扩展区域,较高的值会产生更小的扩展区域。

**Q: 羽化功能如何使用?**
A: 将 `Mask Feathering` 节点连接到 `Mask Expansion` 节点之后,设置合适的feather_size即可。

**Q: 能否组合使用不同的扩展模式?**
A: 可以串联多个 `Mask Expansion` 节点,使用不同的模式来实现更复杂的效果。

## 依赖项

```
torch>=1.9.0
numpy>=1.19.0
scipy>=1.5.0
```

## 许可证

本项目遵循ComfyUI的许可证。

## 贡献

欢迎提交问题和改进建议!
