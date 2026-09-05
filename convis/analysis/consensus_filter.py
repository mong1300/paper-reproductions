"""§7 후보 캡션 합의를 사후 필터로 적용했을 때의 CHAIR.

후보 캡션 n개 중 k개 미만에만 등장한 물체를 최종 캡션에서 제거하고 CHAIR 를 다시 계산한다.
T2I 생성도 대조 디코딩도 사용하지 않는다.

주의: kmin=0 이 필터 없음(원본)이다. kmin=1 로 두면 후보 캡션 어디에도 없던 물체가
      이미 제거되어 원본과 값이 달라진다.

검증값 (greedy 캡션 기준):
    kmin=0   CHAIR_S 19.0   CHAIR_I 8.4   환각 25   언급 296   제거   0   그중 환각  -
    kmin=1           13.0            5.2        15        286        10             10 (100%)
    kmin=2            7.0            3.3         9        274        22             16  (73%)
    kmin=3            5.0            2.7         7        264        32             18  (56%)
    kmin=4            2.0            0.9         2        229        67             23  (34%)
"""


def apply_filter(alpha, agree, kmin):
    """k < kmin 인 물체를 제거하고 (CHAIR_S, CHAIR_I, 환각 수, 언급 수,
    제거된 언급 수, 그중 환각이었던 수) 를 반환한다.

    CHAIR_S = 환각이 남은 이미지 비율
    CHAIR_I = 남은 환각 수 / 남은 언급 수
    """
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
