import argparse
import torch
import pdb
import json
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    input_path = args.infile_path
    output_path = args.ourfile_path
    img_source_path = args.img_source
    lora_path = args.lora_adapt

    similar_img_path = "v-rag similar images json path"
    similar_img_source_path = "v-rag similar images path"
    similar_report = "v-rag similar report path"

    with open(input_path, encoding='utf-8') as f:
        struct = json.load(f)
    
    with open(similar_img_path, encoding='utf-8') as f1:
        similar_json = json.load(f1) 

    with open(similar_report, encoding='utf-8') as f2:
        similar_report_json = json.load(f2)   

    save_count = 0
    top_k = 1
    for patient in tqdm(struct): 
        save_count += 1
        for report in struct[patient]:
            images =  struct[patient][report]['images']
            ents = struct[patient][report]['predicted_ner']
            ent_types = struct[patient][report]['predicted_types']
            struct[patient][report]["imgprobing"] = {}
            if len(images) == 1:
                img = images[0]
                top_k_similar_images = similar_json[img[:-4]][:top_k]
                img_prob_list = []
                for entity, ent_type in zip(ents, ent_types):
                    problem = entity.lower()
                    image_load_path = img_source_path + patient[:3] + '/' + patient + '/' + report + '/' + img


                    question = ''
                    image = []
                    img_count = 0
                    for sim_img in top_k_similar_images:
                        imgid = sim_img[sim_img.rfind('/')+1:]
                        sim_report = similar_report_json[imgid].replace('\n', '').strip()
                        sim_path = similar_img_source_path + sim_img[:3] + '/' + sim_img + '.jpg'
                        image = image + [sim_path]
                        question += '<image-placeholder> This is the {} similar image and its report for your reference. Report: {} '.format(str(img_count+1), sim_report)
                        img_count += 1
                        
                    question += '<image-placeholder>  Answer the question with only the word yes or no. According to the last query image and the reference images and reports, does the patient have ' + problem + '?' 
                    
                    image = image + [image_load_path]
                    

                    qs = question

                    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                    if IMAGE_PLACEHOLDER in qs:
                        if model.config.mm_use_im_start_end:
                            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                        else:
                            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
                    else:
                        if model.config.mm_use_im_start_end:
                            qs = image_token_se + "\n" + qs
                        else:
                            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                    if "llama-2" in model_name.lower():
                        conv_mode = "llava_llama_2"
                    elif "mistral" in model_name.lower():
                        conv_mode = "mistral_instruct"
                    elif "v1.6-34b" in model_name.lower():
                        conv_mode = "chatml_direct"
                    elif "v1" in model_name.lower():
                        conv_mode = "llava_v1"
                    elif "mpt" in model_name.lower():
                        conv_mode = "mpt"
                    else:
                        conv_mode = "llava_v0"

                    if args.conv_mode is not None and conv_mode != args.conv_mode:
                        print(
                            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                                conv_mode, args.conv_mode, args.conv_mode
                            )
                        )
                    else:
                        args.conv_mode = conv_mode

                    conv = conv_templates[args.conv_mode].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    image_files = image # a list of images' paths

                    images = load_images(image_files)
                    image_sizes = [x.size for x in images]
                    images_tensor = process_images(
                        images,
                        image_processor,
                        model.config
                    ).to(model.device, dtype=torch.float16)
                    input_ids = (
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .cuda()
                    )

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=images_tensor,
                            image_sizes=image_sizes,
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                        )

                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    
                    if re.search(r"Yes", outputs, flags=re.IGNORECASE):
                        img_prob_list.append('Yes')
                    elif re.search(r"No", outputs, flags=re.IGNORECASE):
                        img_prob_list.append('No')
                    else:
                        img_prob_list.append(outputs)
                struct[patient][report]["imgprobing"][img] = img_prob_list

        if save_count > 0 and save_count % 50 == 0:       
            print("save: ", save_count) 
            with open(output_path + 'test-ncbi-radfm_prob_vRAG_addimages_withreport_finetune_' + lora_path.split('/')[-1] + '_top' + str(top_k) , "w") as outfile: # withreport/negative need add
                outfile.write(json.dumps(struct, indent=2)) 
    with open(output_path + 'test-ncbi-radfm_prob_vRAG_addimages_withreport_finetune_' + lora_path.split('/')[-1] + '_top' + str(top_k) , "w") as outfile: # withreport/negative need add
        outfile.write(json.dumps(struct, indent=2)) 
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--infile_path", type=str, required=True)
    parser.add_argument("--ourfile_path", type=str, required=True)
    parser.add_argument("--img_source", type=str, required=True)
    parser.add_argument("--lora_adapt", type=str, required=True)

    args = parser.parse_args()

    eval_model(args)
