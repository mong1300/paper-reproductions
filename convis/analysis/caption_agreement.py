"""§3 후보 캡션 간 합의 수준과 최종 캡션에서의 생존율.

후보 캡션 n개가 각각 어떤 COCO 물체를 언급했는지 세어, 물체별 합의 수준 k (1~n) 를 구한다.
그 다음 각 조건의 최종 캡션에서 k 별 생존율과, 생존한 물체 중 환각인 비율을 집계한다.

검증값 (greedy, n=4):
    k=4/4   물체 143개   생존 81%   생존 시 환각 1.7%
    k=3/4        67개        40%              11.1%
    k=2/4        59개        17%              20.0%
    k=1/4       133개         9%              50.0%
    언급된 물체의 48% 가 4개 중 하나의 캡션에만 등장
"""


def agreement_counts():
    """이미지별로 {물체: 언급한 캡션 수} 를 반환한다.

    캡션 하나 안에서의 중복은 세지 않는다 (set 으로 축약한 뒤 누적).
    """
    raise NotImplementedError


def survival_by_k(alpha, agree):
    """조건 alpha 의 최종 캡션에서, k 수준별 생존율과 환각률을 집계한다."""
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
