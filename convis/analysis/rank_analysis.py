"""§2 대조 디코딩이 개입하는 위치.

각 조건이 만든 캡션을 원본 이미지와 함께 teacher-forcing 으로 통과시켜,
출력된 토큰이 f(v) 기준 몇 순위였는지 측정한다. 순위가 1 이 아니면 대조가 뒤집은 것이다.
캡션당 forward 1회면 되므로 디코딩 루프를 다시 돌 필요가 없다.

모호성은 f(v) 의 1위-2위 확률 차이(gap)로 정의하고, 구간별로 집계한다.

구현 주의:
    프롬프트 뒤 타깃 토큰 구간의 로짓만 써야 한다. 길이 n 의 타깃이면 logits[-n-1:-1] 이
    각 타깃 토큰 '직전' 위치에 해당한다.

검증값 (gap 구간별 '1위 아님' 비율):
             <0.1   0.1-0.3   0.3-0.6   >0.6
    a=-1     52.0%    26.8%     11.4%   0.7%
    a=-0.5   39.0%    14.3%      2.8%   0.0%
    a=-0.1   13.1%     0.3%      0.0%   0.0%
    a= 0.1   12.7%     0.5%      0.1%   0.0%
    a= 0.5   35.9%    11.9%      3.6%   0.2%
    a= 1     47.8%    29.5%     10.4%   0.6%

    뒤집혔을 때의 순위 (a=1, 가장 애매한 구간): 2위 58% / 3위 21% / 4위 이상 21%
"""

BINS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)]


def token_ranks(model, processor, image, caption, prompt):
    """caption 을 image 와 함께 teacher-forcing 하고,
    각 위치의 (순위, top1-top2 확률차) 리스트를 반환한다."""
    raise NotImplementedError


def flip_rate_by_bin(alpha):
    """조건 alpha 에 대해 gap 구간별 '1위 아님' 비율과 순위 분포를 집계한다."""
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
