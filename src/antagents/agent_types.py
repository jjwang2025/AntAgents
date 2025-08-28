# coding=utf-8

import logging
import os
import pathlib
import tempfile
import uuid
from io import BytesIO
from typing import Any

import PIL.Image
import requests

from .utils import _is_package_available


logger = logging.getLogger(__name__)


class AgentType:
    """
    抽象类，需要重新实现以定义可由智能体返回的类型。

    这些对象有三个用途：

    - 它们的行为类似于它们所代表的类型，例如文本对应字符串，图像对应 PIL.Image.Image
    - 它们可以被字符串化：str(object) 以返回定义该对象的字符串
    - 它们应该在 ipython notebooks/colab/jupyter 中正确显示
    """

    def __init__(self, value):
        self._value = value

    def __str__(self):
        return self.to_string()

    def to_raw(self):
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return self._value

    def to_string(self) -> str:
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return str(self._value)


class AgentText(AgentType, str):
    """
    智能体返回的文本类型。行为类似于字符串。
    """

    def to_raw(self):
        return self._value

    def to_string(self):
        return str(self._value)


class AgentImage(AgentType, PIL.Image.Image):
    """
    智能体返回的图像类型。行为类似于 PIL.Image.Image。
    """

    def __init__(self, value):
        AgentType.__init__(self, value)
        PIL.Image.Image.__init__(self)

        self._path = None
        self._raw = None
        self._tensor = None

        if isinstance(value, AgentImage):
            self._raw, self._path, self._tensor = value._raw, value._path, value._tensor
        elif isinstance(value, PIL.Image.Image):
            self._raw = value
        elif isinstance(value, bytes):
            self._raw = PIL.Image.open(BytesIO(value))
        elif isinstance(value, (str, pathlib.Path)):
            self._path = value
        else:
            try:
                import torch

                if isinstance(value, torch.Tensor):
                    self._tensor = value
                import numpy as np

                if isinstance(value, np.ndarray):
                    self._tensor = torch.from_numpy(value)
            except ModuleNotFoundError:
                pass

        if self._path is None and self._raw is None and self._tensor is None:
            raise TypeError(f"Unsupported type for {self.__class__.__name__}: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        """
        在 ipython notebook（ipython, colab, jupyter, ...）中正确显示此类型
        """
        from IPython.display import Image, display

        display(Image(self.to_string()))

    def to_raw(self):
        """
        返回该对象的"原始"版本。对于 AgentImage，它是一个 PIL.Image.Image。
        """
        if self._raw is not None:
            return self._raw

        if self._path is not None:
            self._raw = PIL.Image.open(self._path)
            return self._raw

        if self._tensor is not None:
            import numpy as np

            array = self._tensor.cpu().detach().numpy()
            return PIL.Image.fromarray((255 - array * 255).astype(np.uint8))

    def to_string(self):
        """
        返回该对象的字符串化版本。对于 AgentImage，它是序列化图像的文件路径。
        """
        if self._path is not None:
            return self._path

        if self._raw is not None:
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".png")
            self._raw.save(self._path, format="png")
            return self._path

        if self._tensor is not None:
            import numpy as np

            array = self._tensor.cpu().detach().numpy()

            # 这里可能有比加载图像再保存更简单的方法
            img = PIL.Image.fromarray((255 - array * 255).astype(np.uint8))

            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".png")
            img.save(self._path, format="png")

            return self._path

    def save(self, output_bytes, format: str = None, **params):
        """
        将图像保存到文件。
        参数:
            output_bytes (bytes): 保存图像的输出字节。
            format (str): 输出图像的格式。格式与 PIL.Image.save 相同。
            **params: 传递给 PIL.Image.save 的额外参数。
        """
        img = self.to_raw()
        img.save(output_bytes, format=format, **params)


class AgentAudio(AgentType, str):
    """
    智能体返回的音频类型。
    """

    def __init__(self, value, samplerate=16_000):
        if not _is_package_available("soundfile") or not _is_package_available("torch"):
            raise ModuleNotFoundError(
                "Please install 'audio' extra to use AgentAudio: `pip install 'antagents[audio]'`"
            )
        import numpy as np
        import torch

        super().__init__(value)

        self._path = None
        self._tensor = None

        self.samplerate = samplerate
        if isinstance(value, (str, pathlib.Path)):
            self._path = value
        elif isinstance(value, torch.Tensor):
            self._tensor = value
        elif isinstance(value, tuple):
            self.samplerate = value[0]
            if isinstance(value[1], np.ndarray):
                self._tensor = torch.from_numpy(value[1])
            else:
                self._tensor = torch.tensor(value[1])
        else:
            raise ValueError(f"Unsupported audio type: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        """
        在 ipython notebook（ipython, colab, jupyter, ...）中正确显示此类型
        """
        from IPython.display import Audio, display

        display(Audio(self.to_string(), rate=self.samplerate))

    def to_raw(self):
        """
        返回该对象的"原始"版本。它是一个 `torch.Tensor` 对象。
        """
        import soundfile as sf

        if self._tensor is not None:
            return self._tensor

        import torch

        if self._path is not None:
            if "://" in str(self._path):
                response = requests.get(self._path)
                response.raise_for_status()
                tensor, self.samplerate = sf.read(BytesIO(response.content))
            else:
                tensor, self.samplerate = sf.read(self._path)
            self._tensor = torch.tensor(tensor)
            return self._tensor

    def to_string(self):
        """
        返回该对象的字符串化版本。对于 AgentAudio，它是序列化音频的文件路径。
        """
        import soundfile as sf

        if self._path is not None:
            return self._path

        if self._tensor is not None:
            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".wav")
            sf.write(self._path, self._tensor, samplerate=self.samplerate)
            return self._path


_AGENT_TYPE_MAPPING = {"string": AgentText, "image": AgentImage, "audio": AgentAudio}


def handle_agent_input_types(*args, **kwargs):
    args = [(arg.to_raw() if isinstance(arg, AgentType) else arg) for arg in args]
    kwargs = {k: (v.to_raw() if isinstance(v, AgentType) else v) for k, v in kwargs.items()}
    return args, kwargs


def handle_agent_output_types(output: Any, output_type: str | None = None) -> Any:
    if output_type in _AGENT_TYPE_MAPPING:
        # 如果类已定义输出，我们可以直接根据类定义映射
        decoded_outputs = _AGENT_TYPE_MAPPING[output_type](output)
        return decoded_outputs

    # 如果类没有定义输出，则根据类型映射
    if isinstance(output, str):
        return AgentText(output)
    if isinstance(output, PIL.Image.Image):
        return AgentImage(output)
    try:
        import torch

        if isinstance(output, torch.Tensor):
            return AgentAudio(output)
    except ModuleNotFoundError:
        pass
    return output


__all__ = ["AgentType", "AgentImage", "AgentText", "AgentAudio"]