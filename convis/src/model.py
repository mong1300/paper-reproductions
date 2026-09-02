import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def load_model(model_id, dtype=torch.float16, device="mps", load_in_4bit=False):
    """VLM 과 processor 로딩.

    load_in_4bit 은 CUDA 전용이다 (bitsandbytes). Colab T4 처럼 VRAM 이 좁은
    환경에서 7B + T2I 를 함께 올리기 위해 쓴다. MPS 에서는 무시된다.
    """
    kwargs = {"torch_dtype": dtype}

    if load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"      # 4bit 는 .to() 로 옮길 수 없다

    model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if "device_map" not in kwargs:
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def generate_caption(model, processor, image, prompt, caption_max_tokens=64, **gen_kwargs):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, model.dtype)

    generate_ids = model.generate(**inputs,
        max_new_tokens=caption_max_tokens,
        **gen_kwargs)

    trimmed = generate_ids[:, inputs["input_ids"].shape[1]:]
    caption = processor.decode(trimmed[0], skip_special_tokens=True).strip()

    return caption


def get_logits(model, cache, input_ids):
    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)

    return out.logits[:, -1, :], out.past_key_values


def convis_decode(model, processor, img, recons, prompt, max_new_tokens, alpha=1.0, lam=0.1):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, model.dtype)

    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    original_logit, original_cache = out.logits[:, -1, :], out.past_key_values

    recon_caches = []
    recon_logits = []
    for recon in recons:
        conversation_recon = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": recon},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs_recon = processor.apply_chat_template(
            conversation_recon,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, model.dtype)

        with torch.no_grad():
            out = model(**inputs_recon, use_cache=True)
        recon_logit, recon_cache = out.logits[:, -1, :], out.past_key_values

        recon_logits.append(recon_logit)
        recon_caches.append(recon_cache)

    generate = []
    eos_ids = model.generation_config.eos_token_id
    for _ in range(max_new_tokens):

        probs = original_logit.softmax(dim=-1)
        mask = probs < lam * probs.max(-1, keepdim=True).values

        logits_mean = torch.stack(recon_logits).mean(0)
        filtered_logit = (1 + alpha) * original_logit - alpha * logits_mean
        filtered_logit = filtered_logit.masked_fill(mask, -float("inf"))
        next_id = filtered_logit.argmax(-1, keepdim=True)

        if next_id.item() == eos_ids:
            break

        generate.append(next_id.item())

        with torch.no_grad():
            out = model(input_ids=next_id, past_key_values=original_cache, use_cache=True)

        original_logit, original_cache =  out.logits[:, -1, :], out.past_key_values

        for i in range(len(recons)):
            with torch.no_grad():
                out = model(input_ids=next_id, past_key_values=recon_caches[i], use_cache=True)

            recon_logits[i], recon_caches[i]  = out.logits[:, -1, :], out.past_key_values

    return processor.decode(generate, skip_special_tokens=True).strip()