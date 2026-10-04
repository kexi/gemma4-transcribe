import { describe, expect, it } from 'vitest';

import { niceScale } from './scale.ts';

describe('niceScale', () => {
  it('0 起点で、最大値を含む切りのよい刻みの目盛りを作る', () => {
    expect(niceScale(38.3)).toEqual({ max: 40, ticks: [0, 10, 20, 30, 40] });
  });

  it('刻みは 1・2・5 × 10^k のどれかになる', () => {
    expect(niceScale(7).ticks).toEqual([0, 2, 4, 6, 8]);
    expect(niceScale(160).ticks).toEqual([0, 50, 100, 150, 200]);
  });

  it('最大値がちょうど目盛りに乗るときは余計に伸ばさない', () => {
    expect(niceScale(40).max).toBe(40);
  });

  it('小数の刻みでも浮動小数の誤差を目盛りに出さない', () => {
    expect(niceScale(0.3).ticks).toEqual([0, 0.1, 0.2, 0.3]);
  });

  it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY])(
    '値が無い・不正（%s）でも 0–1 の軸を返して描画を止めない',
    (value) => {
      expect(niceScale(value)).toEqual({ max: 1, ticks: [0, 1] });
    },
  );

  it('目盛りはいつも max で終わり、max は最大値以上', () => {
    for (const value of [0.3, 3.7, 12.5, 55, 99.9, 120, 999]) {
      const { max, ticks } = niceScale(value);
      expect(ticks.at(-1)).toBe(max);
      expect(max).toBeGreaterThanOrEqual(value);
    }
  });
});
