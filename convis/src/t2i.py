from diffusers import AutoPipelineForText2Image
from compel import Compel

def load_t2i(dtype, device, model_id="stabilityai/sd-turbo"):
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype, variant="fp16")
    pipe = pipe.to(device)
    pipe.compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    return pipe


def regenerate(pipe, caption, num_steps=1):
    embeds = pipe.compel(caption)
    recon = pipe(prompt_embeds=embeds, num_inference_steps=num_steps, guidance_scale=0.0).images[0]
    return recon

