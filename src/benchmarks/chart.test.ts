import { describe, expect, it } from 'vitest';

import { barValueLabel, type BarDatum } from './chart.ts';

const bar = (value: number | null, failed: number | null): BarDatum => ({
  modelKey: 'e2b',
  modelLabel: 'Gemma 4 E2B（q4f16）',
  value,
  n: 100,
  failed,
  median: null,
  outliers: null,
});

describe('barValueLabel', () => {
  it('値を小数 1 桁の % で出す', () => {
    expect(barValueLabel(bar(20.54, 0))).toEqual({ value: '20.5%', note: '' });
  });

  it('失敗クリップがあれば棒の横に「失敗 n 件」を必ず併記する', () => {
    expect(barValueLabel(bar(38.3, 2))).toEqual({ value: '38.3%', note: '失敗 2 件' });
  });

  it('値が無いモデルは「未計測」と出す（棒は描かない）', () => {
    expect(barValueLabel(bar(null, null))).toEqual({ value: '未計測', note: '' });
  });

  it('全件失敗で値が無いときは未計測と区別して「計測不可」と失敗数を出す', () => {
    expect(barValueLabel(bar(null, 100))).toEqual({ value: '計測不可', note: '失敗 100 件' });
  });
});
