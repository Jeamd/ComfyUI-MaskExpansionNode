"""
ComfyUI自定义节点 - Mask Expansion
将严格遮罩转换为区域遮罩
"""

import torch
import numpy as np
from typing import Optional


class MaskExpansionNode:
    """
    将严格遮罩转换为区域遮罩
    支持多种扩展方式:固定padding、按比例扩展、形态学膨胀等
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "mode": (["fixed_padding", "ratio_expand", "morphology", "gaussian_blur"], {
                    "default": "fixed_padding"
                }),
            },
            "optional": {
                "padding": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "expand_ratio": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05
                }),
                "kernel_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 31,
                    "step": 2
                }),
                "blur_sigma": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 20.0,
                    "step": 0.5
                }),
                "blur_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("expanded_masks",)
    FUNCTION = "expand_masks"
    CATEGORY = "mask/transform"
    
    def expand_masks(self, masks, mode="fixed_padding", padding=10, 
                    expand_ratio=0.2, kernel_size=5, blur_sigma=2.0, blur_threshold=0.3):
        """
        扩展masks,从严格遮罩转换为区域遮罩
        
        Args:
            masks: 输入的masks, shape为(N, H, W)的tensor
            mode: 扩展模式
                - fixed_padding: 固定padding扩展
                - ratio_expand: 按比例扩展
                - morphology: 形态学膨胀
                - gaussian_blur: 高斯模糊扩展
            padding: 固定padding值(像素)
            expand_ratio: 扩展比例
            kernel_size: 形态学操作的核大小(奇数)
            blur_sigma: 高斯模糊的标准差
            blur_threshold: 模糊后的二值化阈值
            
        Returns:
            expanded_masks: 扩展后的masks
        """
        
        # 确保输入是torch tensor
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)
        
        # 处理不同的扩展模式
        if mode == "fixed_padding":
            expanded_masks = self._expand_with_padding(masks, padding)
        elif mode == "ratio_expand":
            expanded_masks = self._expand_with_ratio(masks, expand_ratio)
        elif mode == "morphology":
            expanded_masks = self._expand_with_morphology(masks, kernel_size)
        elif mode == "gaussian_blur":
            expanded_masks = self._expand_with_gaussian(masks, blur_sigma, blur_threshold)
        else:
            expanded_masks = masks
        
        return (expanded_masks,)
    
    def _expand_with_padding(self, masks, padding):
        """
        使用固定padding扩展mask
        
        Args:
            masks: 输入masks
            padding: padding值
            
        Returns:
            扩展后的masks
        """
        if padding == 0:
            return masks
        
        # 转换为numpy进行操作
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        
        if len(masks_np.shape) == 2:
            masks_np = masks_np[np.newaxis, :, :]
        
        expanded_masks = []
        for mask in masks_np:
            # 计算mask的边界
            rows = np.any(mask > 0.5, axis=1)
            cols = np.any(mask > 0.5, axis=0)
            
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            
            if len(y_indices) > 0 and len(x_indices) > 0:
                y_min, y_max = y_indices[0], y_indices[-1]
                x_min, x_max = x_indices[0], x_indices[-1]
                
                # 创建扩展后的mask
                height, width = mask.shape
                expanded_mask = np.zeros_like(mask)
                
                # 计算扩展后的边界
                y_min_exp = max(0, y_min - padding)
                y_max_exp = min(height, y_max + padding + 1)
                x_min_exp = max(0, x_min - padding)
                x_max_exp = min(width, x_max + padding + 1)
                
                # 设置扩展区域为1
                expanded_mask[y_min_exp:y_max_exp, x_min_exp:x_max_exp] = 1.0
                
                # 保留原始mask区域
                expanded_mask = np.maximum(expanded_mask, mask)
                
                expanded_masks.append(expanded_mask)
            else:
                expanded_masks.append(mask.copy())
        
        result = torch.from_numpy(np.stack(expanded_masks)).float()
        return result
    
    def _expand_with_ratio(self, masks, ratio):
        """
        按比例扩展mask
        
        Args:
            masks: 输入masks
            ratio: 扩展比例
            
        Returns:
            扩展后的masks
        """
        if ratio <= 0:
            return masks
        
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        
        if len(masks_np.shape) == 2:
            masks_np = masks_np[np.newaxis, :, :]
        
        expanded_masks = []
        for mask in masks_np:
            # 计算mask的边界
            rows = np.any(mask > 0.5, axis=1)
            cols = np.any(mask > 0.5, axis=0)
            
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            
            if len(y_indices) > 0 and len(x_indices) > 0:
                y_min, y_max = y_indices[0], y_indices[-1]
                x_min, x_max = x_indices[0], x_indices[-1]
                
                # 计算原始尺寸
                height = y_max - y_min + 1
                width = x_max - x_min + 1
                
                # 计算扩展量
                expand_h = int(height * ratio)
                expand_w = int(width * ratio)
                
                # 创建扩展后的mask
                h, w = mask.shape
                expanded_mask = np.zeros_like(mask)
                
                # 计算扩展后的边界
                y_min_exp = max(0, y_min - expand_h)
                y_max_exp = min(h, y_max + expand_h + 1)
                x_min_exp = max(0, x_min - expand_w)
                x_max_exp = min(w, x_max + expand_w + 1)
                
                # 设置扩展区域为1
                expanded_mask[y_min_exp:y_max_exp, x_min_exp:x_max_exp] = 1.0
                
                # 保留原始mask区域
                expanded_mask = np.maximum(expanded_mask, mask)
                
                expanded_masks.append(expanded_mask)
            else:
                expanded_masks.append(mask.copy())
        
        result = torch.from_numpy(np.stack(expanded_masks)).float()
        return result
    
    def _expand_with_morphology(self, masks, kernel_size):
        """
        使用形态学膨胀扩展mask
        
        Args:
            masks: 输入masks
            kernel_size: 膨胀核大小
            
        Returns:
            扩展后的masks
        """
        from scipy.ndimage import binary_dilation
        
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        
        if len(masks_np.shape) == 2:
            masks_np = masks_np[np.newaxis, :, :]
        
        # 创建方形结构元素
        structure = np.ones((kernel_size, kernel_size), dtype=bool)
        
        expanded_masks = []
        for mask in masks_np:
            # 二值化
            binary_mask = (mask > 0.5)
            
            # 形态学膨胀
            dilated_mask = binary_dilation(binary_mask, structure=structure)
            
            # 转换回浮点数
            expanded_mask = dilated_mask.astype(np.float32)
            
            expanded_masks.append(expanded_mask)
        
        result = torch.from_numpy(np.stack(expanded_masks)).float()
        return result
    
    def _expand_with_gaussian(self, masks, sigma, threshold):
        """
        使用高斯模糊扩展mask
        
        Args:
            masks: 输入masks
            sigma: 高斯模糊标准差
            threshold: 二值化阈值
            
        Returns:
            扩展后的masks
        """
        from scipy.ndimage import gaussian_filter
        
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        
        if len(masks_np.shape) == 2:
            masks_np = masks_np[np.newaxis, :, :]
        
        expanded_masks = []
        for mask in masks_np:
            # 高斯模糊
            blurred = gaussian_filter(mask, sigma=sigma)
            
            # 二值化
            binary_mask = (blurred > threshold).astype(np.float32)
            
            # 保留原始mask区域
            expanded_mask = np.maximum(binary_mask, mask)
            
            expanded_masks.append(expanded_mask)
        
        result = torch.from_numpy(np.stack(expanded_masks)).float()
        return result


class MaskFeatheringNode:
    """
    对mask边缘进行羽化处理,使mask更柔和
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "feather_size": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 50,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("feathered_masks",)
    FUNCTION = "feather_masks"
    CATEGORY = "mask/transform"
    
    def feather_masks(self, masks, feather_size=5):
        """
        对mask边缘进行羽化处理
        
        Args:
            masks: 输入的masks
            feather_size: 羽化大小(像素)
            
        Returns:
            feathered_masks: 羽化后的masks
        """
        if feather_size <= 0:
            return masks
        
        from scipy.ndimage import binary_erosion, binary_dilation
        
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        
        if len(masks_np.shape) == 2:
            masks_np = masks_np[np.newaxis, :, :]
        
        feathered_masks = []
        for mask in masks_np:
            # 二值化
            binary_mask = (mask > 0.5)
            
            if feather_size > 0:
                # 创建腐蚀后的mask
                eroded_mask = binary_erosion(binary_mask, iterations=feather_size)
                
                # 创建膨胀后的mask
                dilated_mask = binary_dilation(binary_mask, iterations=feather_size)
                
                # 计算羽化区域
                feather_region = dilated_mask & ~eroded_mask
                
                # 创建平滑的羽化mask
                from scipy.ndimage import distance_transform_edt
                distance_out = distance_transform_edt(~dilated_mask)
                distance_in = distance_transform_edt(eroded_mask)
                
                # 创建羽化权重
                feather_mask = np.zeros_like(mask, dtype=np.float32)
                feather_mask[binary_mask] = 1.0
                
                if np.any(feather_region):
                    # 在羽化区域创建平滑过渡
                    total_dist = distance_out[feather_region] + distance_in[feather_region] + 1e-8
                    feather_mask[feather_region] = distance_in[feather_region] / total_dist
                
                feathered_masks.append(feather_mask)
            else:
                feathered_masks.append(binary_mask.astype(np.float32))
        
        result = torch.from_numpy(np.stack(feathered_masks)).float()
        return result


class MaskInvertNode:
    """
    反转mask(黑白互换)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("inverted_masks",)
    FUNCTION = "invert_masks"
    CATEGORY = "mask/transform"
    
    def invert_masks(self, masks):
        """
        反转mask
        
        Args:
            masks: 输入的masks
            
        Returns:
            inverted_masks: 反转后的masks
        """
        return (1.0 - masks,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "MaskExpansion": MaskExpansionNode,
    "MaskFeathering": MaskFeatheringNode,
    "MaskInvert": MaskInvertNode,
}

# 节点名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskExpansion": "Mask Expansion (严格遮罩到区域遮罩)",
    "MaskFeathering": "Mask Feathering (边缘羽化)",
    "MaskInvert": "Mask Invert (反转遮罩)",
}
