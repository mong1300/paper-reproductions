"""§4 환각이 발생한 COCO 카테고리의 분포.

조건별로 어떤 카테고리가 환각되는지 세고, COCO val2014 전체에서의 출현 빈도(기저율)와 비교한다.
기저율은 인스턴스 수가 아니라 '해당 카테고리가 등장하는 이미지의 비율' 로 계산한다.

검증값 (greedy, 환각 25건 / 12종):
    orange          4 / 4 언급  (100%)   기저율 1.4%
    knife           4 / 4       (100%)          3.5%
    dining table    4 / 27       (15%)          9.9%
    tv              3 / 5        (60%)          3.9%
    최빈 10개 카테고리가 차지하는 환각 비율 28%
    person (최빈 카테고리) 환각 0건
"""


def coco_base_rates():
    """data/coco/instances_val2014.json 에서 카테고리별 출현 이미지 비율을 계산한다.

    어노테이션을 image_id 로 묶어 이미지당 카테고리 집합을 만든 뒤,
    각 카테고리가 몇 개 이미지에 등장하는지 세고 전체 이미지 수로 나눈다.
    """
    raise NotImplementedError


def category_stats(alpha):
    """조건 alpha 의 환각 카테고리별 (환각 수, 언급 수, 환각률) 을 반환한다."""
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
