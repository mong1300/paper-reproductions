from diffusers import AutoPipelineForText2Image

def load_t2i(dtype, device, model_id="stabilityai/sd-turbo"):
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype, variant="fp16")
    pipe = pipe.to(device)

    return pipe


def regenerate(pipe, caption, num_steps=1):
    recon = pipe(prompt=caption, num_inference_steps=num_steps, guidance_scale=0.0).images[0]
    return recon

