"""VLM (LLaVA 등) 로딩과 캡션 생성.

이 파일은 '정의'만 한다. import 해도 아무 일도 일어나지 않아야 한다.
(전역에서 모델을 로딩하지 말 것)
"""


def load_model(model_id, dtype="float16", device="auto"):
    """모델과 processor를 로딩해서 반환.

    Args:
        model_id: HuggingFace 모델 id (예: "llava-hf/llava-1.5-7b-hf")
        dtype: "float16" | "float32"
        device: "auto" | "cuda" | "mps" | "cpu"

    Returns:
        (model, processor)
    """
    # TODO: transformers 로 모델/processor 로딩
    # TODO: CUDA면 4-bit 양자화 고려, MPS면 불가하므로 분기
    raise NotImplementedError


def generate_caption(model, processor, image, prompt, max_new_tokens=64):
    """이미지 한 장 -> 캡션 문자열.

    Stage 0~1에서 쓰는 평범한(baseline) 생성.

    Returns:
        str
    """
    # TODO: processor 로 입력 만들고 model.generate 호출
    # TODO: decode 후 프롬프트 부분 잘라내기
    raise NotImplementedError


def get_logits(model, processor, image, input_ids):
    """이미지 + 지금까지 생성된 토큰 -> 다음 토큰의 로짓 분포.

    Stage 2(ConVis)에서 원본/재구성 이미지 각각에 대해 호출된다.
    generate_caption 과 달리 한 스텝만 진행한다.

    Returns:
        Tensor, shape (vocab_size,)
    """
    # TODO: model(...) 을 직접 호출해서 logits[:, -1, :] 반환
    raise NotImplementedError
