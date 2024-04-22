from torchtune.models.llama3 import qlora_llama3_8b,llama3_tokenizer
import torch
from torchtune.modules.peft import LoRALinear

torch.set_default_device("cuda")
qlora_linear = LoRALinear(512, 512, rank=256, alpha=0.1, quantize_base=True)
print(torch.cuda.memory_allocated())  # 177,152 bytes

qlora_model = qlora_llama3_8b(lora_attn_modules=["q_proj", "v_proj"])

attn = qlora_model.layers[0].attn
print(type(attn.q_proj.weight))  # <class 'torchao.dtypes.nf4tensor.NF4Tensor'>
print(type(attn.k_proj.weight))  # <class 'torch.nn.parameter.Parameter'>