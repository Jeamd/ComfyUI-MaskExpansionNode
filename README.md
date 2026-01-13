# ComfyUI Mask to Area Mask 自定义节点

这是一个ComfyUI自定义节点,用于将精准遮罩(严格遮罩)转换为区域遮罩。

## 功能特性

### Mask to Area Mask (精准遮罩转区域遮罩)
将输入的精准遮罩转换为区域遮罩,通过检测mask边界并扩展来创建区域遮罩。

**输入参数:**
- `masks`: 输入的遮罩(MASK类型)
- `padding`: 扩展padding值(像素) - 0-100, 默认20

**输出:**
- `area_masks`: 区域遮罩

## 安装方法

1. 将 `MaskExpansionNode` 文件夹复制到你的ComfyUI自定义节点目录:
   ```
   ComfyUI/custom_nodes/MaskExpansionNode/
   ```

2. 重启ComfyUI,节点将自动加载。

## 使用示例

### 基本用法流程

```
Mask Input → Mask to Area Mask → 后续处理
```

### 使用示例

1. 输入一个精准的分割mask(例如SAM生成的精确轮廓)
2. 连接到 `Mask to Area Mask` 节点
3. 调整 `padding` 参数控制扩展范围
4. 输出扩展后的区域遮罩

**参数说明:**
- `padding=0`: 不扩展,保持原mask
- `padding=10`: 向外扩展10像素
- `padding=20`: 向外扩展20像素(默认值)
- `padding=50`: 向外扩展50像素

**适用场景:**
- 将SAM等分割模型的精准轮廓转换为区域遮罩
- 为对象添加周围的上下文区域
- 扩展mask范围以包含更多背景信息

## 工作原理

### Mask to Area Mask 节点

1. **检测边界**: 找到mask中所有非零像素的最小外接矩形
2. **计算扩展**: 根据padding参数向外扩展矩形边界
3. **生成区域mask**: 创建填充了扩展矩形的区域遮罩
4. **边界处理**: 确保扩展后的边界不超出图像范围

## 技术细节

- 支持批量处理多个mask
- 自动处理torch.Tensor和numpy格式
- 边界检查确保扩展不超出图像范围
- 使用float32类型保持精度

## 常见问题

**Q: 节点不生效怎么办?**
A: 
1. 确保ComfyUI已重启
2. 检查节点是否出现在 `mask/transform` 分类下
3. 确认输入的mask是MASK类型
4. 检查padding参数是否大于0

**Q: 如何控制扩展范围?**
A: 通过调整 `padding` 参数来控制扩展范围。值越大,扩展区域越大。

**Q: 可以处理多个mask吗?**
A: 可以,节点支持批量处理多个mask。

**Q: 扩展后的边界超出图像范围怎么办?**
A: 节点会自动处理边界,确保扩展后的mask不会超出图像范围。

## 依赖项

```
torch>=1.9.0
numpy>=1.19.0
```

## 许可证

本项目遵循ComfyUI的许可证。


## 贡献

欢迎提交问题和改进建议!
