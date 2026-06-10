"""Draw 模块 —— BaseDrawer 抽象基类."""
from abc import ABC, abstractmethod


class BaseDrawer(ABC):
    """文生图统一接口.

    所有生图适配器（豆包、通义千问等）必须实现 generate()。
    """

    model_name: str = "base"

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """生成图片，返回本地文件路径.

        Args:
            prompt: 自然语言图像描述.

        Returns:
            str: 生成图片的本地存储路径.
        """
        ...
