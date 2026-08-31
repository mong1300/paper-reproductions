import yaml
import random

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """random / numpy / torch 시드를 한 번에 고정."""
    random.seed(seed)


def save_jsonl(path, record):
    """결과 한 건을 append. Colab 세션이 끊겨도 여기까지는 남는다."""
    # TODO
    raise NotImplementedError


def load_done_ids(path):
    """이미 처리된 id 집합. 이어서 실행할 때 건너뛰기용."""
    # TODO
    raise NotImplementedError
