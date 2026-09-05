"""§5 CHAIR 가 환각으로 집계한 항목의 추출과 오탐 판정 집계.

두 가지 일을 한다.
  1) 전 조건의 환각 항목을 추출해 LLM 판정 입력을 만든다
  2) 판정 결과(results/fp_review.json)를 조건별로 집계한다

표층 단어(surface)를 반드시 함께 기록해야 한다. 동의어 사전을 거치면
캡션에 실제로 등장한 단어와 집계된 카테고리명이 달라지기 때문이다
(television -> tv, seat -> chair, container -> bowl, Turkey -> bird).

판정 범주:
    A1 색상 형용사    A2 동음이의어   A3 고유명사
    A4 동의어 과확장  A5 부분-전체    A6 묘사 대상 오인
    B1 어노테이션 누락   C1 실제 환각   D1 판단 불가

검증값:
    전체 164건
    A 99건 (A1 39 / A4 28 / A2 18 / A6 6 / A5 4 / A3 4)
    B1 18건   C1 22건   D1 25건
    조건별 오탐률 (A+B) 55~90%
"""


def extract_items():
    """전 조건에서 환각 항목을 추출한다.

    항목당 필드: alpha, image_id, caption, flagged, surface, gt_objects
    """
    raise NotImplementedError


def aggregate_review():
    """results/fp_review.json 을 읽어 조건별 A/B/C/D 분류 수와 오탐률을 집계한다."""
    raise NotImplementedError


def corrected_chair(criterion):
    """오탐을 제외한 CHAIR 를 계산한다.

    criterion="conservative"  A1, A2, A3, A6 만 제외 (명백한 어휘 오류)
    criterion="strict"        C1 만 환각으로 계산

    검증값 (conservative):  greedy 11.0 / 5.7,  alpha=1  6.0 / 3.2
             (strict):      greedy  3.0 / 1.4,  alpha=1  1.0 / 0.3
    """
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
