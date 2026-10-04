import { describe, expect, it } from 'vitest';

import { expectedClipCount, incompleteStatus, isComparable } from './compare.ts';
import { fixtureResults, metrics, model, withModels } from './testing.ts';

describe('expectedClipCount', () => {
  it('datasets の n の合計を全クリップ数とする（フィクスチャは 100 × 3）', () => {
    expect(expectedClipCount(fixtureResults())).toBe(300);
  });

  it('datasets の n が欠けていれば、計測済みモデルの最大の n で代える', () => {
    const results = {
      ...withModels([model('a', 'A', metrics(0.1, 0.1, { n: 90 })), model('b', 'B', metrics(0.1, 0.1, { n: 120 }))]),
      datasets: fixtureResults().datasets.map((dataset) => ({ ...dataset, n: null })),
    };
    expect(expectedClipCount(results)).toBe(120);
  });
});

describe('incompleteStatus / isComparable', () => {
  it('未実行のクリップがあれば件数を言い、比較に入れない', () => {
    const partial = model('a', 'A', metrics(0.1, 0.1, { n: 120 }), { missing: 180 });
    expect(incompleteStatus(partial, 300)).toBe('未実行 180 件');
    expect(isComparable(partial, 300)).toBe(false);
  });

  it('missing が 0 でも件数が全クリップ数と違えば、比較に入れない', () => {
    const short = model('a', 'A', metrics(0.1, 0.1, { n: 250 }));
    expect(incompleteStatus(short, 300)).toBe('300 クリップ中 250 クリップ');
    expect(isComparable(short, 300)).toBe(false);
  });

  it('全クリップを走り終えたモデルは、失敗があっても比較に入れる（失敗は値の横に注記する）', () => {
    const done = model('a', 'A', metrics(0.1, 0.1, { failed: 2 }));
    expect(incompleteStatus(done, 300)).toBeNull();
    expect(isComparable(done, 300)).toBe(true);
  });

  it('missing の項目が無い古い JSON（null）や件数が不明なモデルは、走り終えたものとして扱う', () => {
    const legacy = model('a', 'A', metrics(0.1, 0.1, { n: null }), { missing: null });
    expect(isComparable(legacy, 300)).toBe(true);
  });

  it('未計測（overall が無い）のモデルは未完了とは言わず、比較にも入れない', () => {
    const unmeasured = model('a', 'A', null, { missing: 300 });
    expect(incompleteStatus(unmeasured, 300)).toBeNull();
    expect(isComparable(unmeasured, 300)).toBe(false);
  });
});
