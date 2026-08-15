import logging

import numpy as np
from transformers import AutoProcessor

class XiaomiTokenizer:
    def __init__(self, max_len: int = 200, qwen_tokenizer_path: str = "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain"):
        self._max_len = max_len
        self._tokenizer = AutoProcessor.from_pretrained(qwen_tokenizer_path, trust_remote_code=True)

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        inputs = self._tokenizer.tokenizer(prompt)
        tokens = inputs['input_ids']

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)
