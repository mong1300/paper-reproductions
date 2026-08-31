"""캡션을 다시 이미지로 되돌리는 T2I (ConVis의 핵심 전제).

Stage 3에서 'T2I 성능이 병목인가'를 검증하려면 모델을 갈아끼워야 하므로
인터페이스를 고정해 둔다.
"""


def load_t2i(model_id="stabilityai/sd-turbo", dtype="float16", device="auto"):
    """T2I 파이프라인 로딩.

    Returns:
        pipeline
    """
    # TODO: diffusers AutoPipelineForText2Image 로 로딩
    raise NotImplementedError


def regenerate(pipe, caption, num_steps=4, seed=42):
    """캡션 -> PIL 이미지.

    seed 를 반드시 고정할 것. 고정하지 않으면 같은 캡션에 매번 다른 이미지가
    나와서 실험 재현이 불가능해진다.

    Returns:
        PIL.Image
    """
    # TODO: generator 에 seed 심고 pipe(caption, ...) 호출
    raise NotImplementedError
