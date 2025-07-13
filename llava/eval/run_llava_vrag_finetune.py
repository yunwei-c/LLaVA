from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_vrag import eval_model


model_path = "/home/mli/ychu/code/LLaVA/ckpt/merge/llava-finetune-adapter"
model_base = "/home/mli/ychu/code/LLaVA/ckpt/merge/base_model"


infile = 'path to input data'
outfile = 'output path'
img_source_path = 'path to image sources'


args = type('Args', (), {
    "model_path": model_path,
    "model_base": model_base,
    "model_name": get_model_name_from_path(model_path),
    "query": None,
    "conv_mode": None,
    "image_file": None,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 20,
    "infile_path":infile,
    "ourfile_path":ourfile,
    "img_source":img_source_path,
    "lora_adapt": model_path
})()

eval_model(args)