from huggingface_hub import snapshot_download,hf_hub_download
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
cache_dir = "./"
cache_dataset_dir = "./"
model_ids = ["stabilityai/stable-diffusion-xl-base-0.9","stabilityai/stable-diffusion-xl-refiner-0.9"]
# dataset_name= ["lambdalabs/pokemon-blip-captions"]
for model_id in model_ids:
    hf_hub_download(repo_id=model_id, filename="sd_xl_base_0.9.safetensors",cache_dir = cache_dir,resume_download = True)
    # StableDiffusionXLPipeline.from_pretrained(model_id, cache_dir = cache_dir,resume_download = True, torch_dtype=torch.float16, variant="fp16",use_safetensors=True)
    # snapshot_download(repo_id=model_id, cache_dir = cache_dataset_dir, repo_type="dataset", resume_download = True)
    # snapshot_download(repo_id=model_id, cache_dir = cache_dir, resume_download = True)
