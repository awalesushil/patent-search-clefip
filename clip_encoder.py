"""Class for CLIP encoder."""

import numpy as np

import torch

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def _node_get(node: torch._C.Node, key: str):
    """Gets attributes of a node which is polymorphic over return type."""
    sel = node.kindOf(key)
    return getattr(node, sel)(key)

torch._C.Node.__getitem__ = _node_get

class CLIPEncoder:
    """Class for CLIP encoder."""

    def __init__(self, model_name: str = "ViT_B_32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model, self.preprocess = torch.hub.load('openai/clip', self.model_name)
        self.model.to(self.device).eval()
        self.warmup()
        self.tokenizer = torch.hub.load('openai/clip', 'tokenize')

    def warmup(self):
        """Warmup model."""
        with torch.inference_mode():
            self.model.encode_image(torch.rand(10, 3, 224, 224).to(self.device))

    def encode(self, inputs: list, text: bool = False) -> torch.Tensor:
        """Encode input."""
        with torch.inference_mode():
            if text:
                return self.model.encode_text(inputs.to(self.device)).detach().cpu().numpy()
            return self.model.encode_image(inputs.to(self.device)).detach().cpu().numpy()
