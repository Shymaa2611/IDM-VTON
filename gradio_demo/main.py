import sys
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer,
)
from diffusers import DDPMScheduler, AutoencoderKL
import torch
import os
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = (binary_mask * 255).astype(np.uint8)
    return Image.fromarray(mask)


base_path = 'yisol/IDM-VTON'
unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float16)
tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer")
tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2")
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=torch.float16)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2", torch_dtype=torch.float16)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_path, subfolder="image_encoder", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16)
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(base_path, subfolder="unet_encoder", torch_dtype=torch.float16)
parsing_model = Parsing(0)
openpose_model = OpenPose(0)
pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)

def start_tryon(human_img_path, garm_img_path, garment_des, is_checked, is_checked_crop, denoise_steps, seed):
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)
    
    human_img = Image.open(human_img_path).convert("RGB").resize((768, 1024))
    garm_img = Image.open(garm_img_path).convert("RGB").resize((768, 1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(human_img.resize((768, 1024)))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt
            )
            pose_img = transforms.ToTensor()(human_img).unsqueeze(0).to(device, torch.float16)
            garm_tensor = transforms.ToTensor()(garm_img).unsqueeze(0).to(device, torch.float16)
            generator = torch.Generator(device).manual_seed(seed) if seed else None
            
            images = pipe(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_img.to(device, torch.float16),
                text_embeds_cloth=prompt_embeds,
                cloth=garm_tensor.to(device, torch.float16),
                mask_image=mask,
                image=human_img, 
                height=1024,
                width=768,
                ip_adapter_image=garm_img.resize((768, 1024)),
                guidance_scale=2.0,
            )[0]
    return images[0], mask


human_img_path = "path/to/human_image.jpg"
garm_img_path = "path/to/garment_image.jpg"
output_image, output_mask = start_tryon(human_img_path, garm_img_path, "Description of garment", True, False, 30, 42)
output_image.show() 
output_mask.show()  
