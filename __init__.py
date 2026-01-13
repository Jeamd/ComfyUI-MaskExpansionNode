"""
ComfyUI自定义节点 - Mask Expansion
将精准遮罩转换为区域遮罩
"""

import torch
import numpy as np


class MaskToAreaMaskNode:
    """
    将精准遮罩转换为区域遮罩
    通过检测mask边界并扩展来创建区域遮罩
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "padding": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
            },
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("area_masks",)
    FUNCTION = "convert_to_area_mask"
    CATEGORY = "mask/transform"
    
    def convert_to_area_mask(self, masks, padding=20):
        """
        将精准遮罩转换为区域遮罩
        
        Args:
            masks: 输入的masks, shape为(N, H, W)的tensor
            padding: 扩展padding值(像素)
            
        Returns:
            area_masks: 区域遮罩
        """
        # 确保输入是torch tensor
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)
        
        if padding == 0:
            return (masks,)
        
        # 转换为numpy进行操作
        masks_np = masks.cpu().numpy()
        
        # 确保是3D格式 (N, H, W)
        if len(masks_np.shape) == 2:
            masks_np = masks_np[np.newaxis, :, :]
        
        area_masks = []
        for mask in masks_np:
            # 找到所有非零像素
            y_coords, x_coords = np.where(mask > 0.5)
            
            if len(y_coords) > 0:
                # 计算边界
                y_min = int(y_coords.min())
                y_max = int(y_coords.max())
                x_min = int(x_coords.min())
                x_max = int(x_coords.max())
                
                height, width = mask.shape
                
                # 创建区域mask
                area_mask = np.zeros_like(mask, dtype=np.float32)
                
                # 计算扩展后的边界
                y_min_exp = max(0, y_min - padding)
                y_max_exp = min(height, y_max + padding + 1)
                x_min_exp = max(0, x_min - padding)
                x_max_exp = min(width, x_max + padding + 1)
                
                # 设置扩展区域为1
                area_mask[y_min_exp:y_max_exp, x_min_exp:x_max_exp] = 1.0
                
                area_masks.append(area_mask)
            else:
                # 如果mask为空,返回原mask
                area_masks.append(mask.copy())
        
        # 转换为torch tensor并返回
        result = torch.from_numpy(np.stack(area_masks)).float()
        return (result,)
    
    # 节点映射
NODE_CLASS_MAPPINGS = {
    "MaskToAreaMask": MaskToAreaMaskNode,
}

# 节点名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskToAreaMask": "Mask to Area Mask (精准遮罩转区域遮罩)",
}
