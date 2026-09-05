"""CHAIR 평가 결과를 읽어오는 공통 로더."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAIR_DIR = ROOT / "outputs/chair"
RESULTS_DIR = ROOT / "results"

ALPHAS = ["-1", "-0.8", "-0.5", "-0.1", "0", "0.1", "0.5", "1"]


def load_chair(name):
    """outputs/chair/{name}_chair_results.json 을 읽어 이미지별 레코드 리스트를 반환한다.

    각 레코드에서 꺼내 쓸 필드:
        image_id
        mscoco_generated_words       언급된 물체 (중복 포함)
        mscoco_hallucinated_words    [표층, 정규화] 쌍의 리스트 — [1] 을 취해야 카테고리명
        mscoco_gt_words              정답 물체
    """
    raise NotImplementedError


def load_captions():
    """outputs/coco/captions.json 을 읽어 이미지별 후보 캡션 n개를 반환한다."""
    raise NotImplementedError


def make_chair_evaluator(image_ids):
    """eval.chair_metrics.chair.CHAIR 인스턴스를 만든다.

    caption_to_words(caption) 의 반환값은 (words, node_words, idxs, double_words) 이며,
    정규화된 COCO 카테고리명은 두 번째 원소다.
    """
    raise NotImplementedError


def save(name, obj):
    """결과를 results/{name}.json 으로 저장한다."""
    raise NotImplementedError
