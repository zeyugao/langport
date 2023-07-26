from typing import List, Optional
from langport.model.executor.base import LocalModelExecutor
from langport.model.model_adapter import get_model_adapter
from llama_cpp import Llama

class LlamaCppTokenizer:
    def __init__(self, model:Llama) -> None:
        self.model = model
    
    def encode(self, text: str) -> List[int]:
        return self.model.tokenize(text.encode())
    
    def decode(self, tokens: List[int]) -> str:
        return self.model.detokenize(tokens).decode()
    
    def is_eos_token(self, token: int) -> bool:
        return self.model.token_eos() == token
        

class LlamaCppExecutor(LocalModelExecutor):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: Optional[str],
        lib: Optional[str] = None,
        gpu_layers: int = 0,
        model_type: str = 'llama',
        chunk_size: int = 1024,
        threads: int = -1,
        quantization: Optional[str] = None,
        cpu_offloading: bool = False,
        n_gqa: int = 1,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super(LlamaCppExecutor, self).__init__(
            model_name = model_name,
            model_path = model_path,
            device = device,
            num_gpus = num_gpus,
            max_gpu_memory = max_gpu_memory,
            quantization = quantization,
            cpu_offloading = cpu_offloading,
        )
        self.gpu_layers = gpu_layers
        # ctransformers has a bug
        self.lib = lib
        self.model_type = model_type
        self.chunk_size = chunk_size
        self.threads = threads
        self.n_gqa = n_gqa
        self.rms_norm_eps = rms_norm_eps
 

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        adapter = get_model_adapter(model_path)
        model = Llama(
            model_path,
            n_gpu_layers=self.gpu_layers,
            n_threads=self.threads,
            n_gqa=self.n_gqa,
            rms_norm_eps=self.rms_norm_eps,
        )
        tokenizer = LlamaCppTokenizer(model)

        return adapter, model, tokenizer
