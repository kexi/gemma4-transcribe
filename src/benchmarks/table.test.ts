import { describe, expect, it } from 'vitest';

import type { BenchmarkResults } from './schema.ts';
import {
  breakdownGroups,
  CACHED_FIRST_LOAD,
  firstLoadText,
  runOutlierSentence,
  summaryTable,
  type SummaryTable,
} from './table.ts';
import { fixtureResults, metrics, model, withModels } from './testing.ts';

/** 見出しで列を引き、太字になった行のモデル key を返す。 */
const bestKeys = (table: SummaryTable, header: string): string[] => {
  const column = table.headers.indexOf(header);
  if (column < 0) throw new Error(`列「${header}」がありません`);
  return table.rows.filter((row) => row.cells[column]?.best === true).map((row) => row.model.key);
};

const cellText = (table: SummaryTable, header: string, key: string): string | undefined => {
  const column = table.headers.indexOf(header);
  return table.rows.find((row) => row.model.key === key)?.cells[column]?.text;
};

describe('summaryTable', () => {
  it('各列の最良値を太字にする（フィクスチャ：総合 CER は E4B、RTF は E2B QAT）', () => {
    const table = summaryTable(fixtureResults());
    expect(bestKeys(table, '総合 正規化')).toEqual(['e4b']);
    expect(bestKeys(table, 'RTF')).toEqual(['e2b-qat']);
  });

  it('未完了のモデルは値が良くても太字にせず、行見出しに比較から外したことを書く', () => {
    const results = withModels([
      model('a', 'A', metrics(0.05, 0.05, { n: 120 }), { missing: 180 }),
      model('b', 'B', metrics(0.2, 0.1)),
      model('c', 'C', metrics(0.3, 0.2)),
    ]);
    const table = summaryTable(results);
    expect(bestKeys(table, '総合 正規化')).toEqual(['b']);
    expect(bestKeys(table, 'RTF')).toEqual(['b']);
    expect(table.rows.map((row) => row.status)).toEqual(['未実行 180 件（未完了のため比較から除外）', null, null]);
  });

  it('件数が全クリップ数に届かないモデルも太字の比較から外す', () => {
    const results = withModels([
      model('a', 'A', metrics(0.05, 0.05, { n: 100 })),
      model('b', 'B', metrics(0.2, 0.1)),
      model('c', 'C', metrics(0.3, 0.2)),
    ]);
    const table = summaryTable(results);
    expect(bestKeys(table, '総合 正規化')).toEqual(['b']);
    expect(table.rows[0]?.status).toBe('300 クリップ中 100 クリップ（未完了のため比較から除外）');
  });

  it('画面で同じ「2.1%」になる 2.15% と 2.1% は両方とも太字にする', () => {
    const results = withModels([model('a', 'A', metrics(0.0215, 0.1)), model('b', 'B', metrics(0.021, 0.2))]);
    expect(bestKeys(summaryTable(results), '総合 正規化')).toEqual(['a', 'b']);
  });

  it('初回読み込み秒の列はキャッシュの有無で意味が変わるので、どのモデルも太字にしない', () => {
    const results = withModels([
      model('a', 'A', metrics(0.1, 0.1), { loadFirstS: 3, firstLoadDownloaded: true }),
      model('b', 'B', metrics(0.2, 0.2), { loadFirstS: 120, firstLoadDownloaded: true }),
      model('c', 'C', metrics(0.3, 0.3), { loadFirstS: 140 }),
    ]);
    expect(bestKeys(summaryTable(results), '読み込み 初回 秒')).toEqual([]);
  });

  it('初回読み込みがキャッシュからだったモデルは秒を出さず「—（キャッシュ済み）」と出す', () => {
    const results = withModels([
      model('a', 'A', metrics(0.1, 0.1), { loadFirstS: 2.8, firstLoadDownloaded: false }),
      model('b', 'B', metrics(0.2, 0.2), { loadFirstS: 120.4, firstLoadDownloaded: true }),
    ]);
    const table = summaryTable(results);
    expect(cellText(table, '読み込み 初回 秒', 'a')).toBe(CACHED_FIRST_LOAD);
    expect(cellText(table, '読み込み 初回 秒', 'b')).toBe('120.4');
  });

  it('失敗が 1 件以上のセルは注意として示す', () => {
    const table = summaryTable(fixtureResults());
    const column = table.headers.indexOf('失敗');
    const warned = table.rows.filter((row) => row.cells[column]?.warning === true).map((row) => row.model.key);
    expect(warned).toEqual(['e2b-qat']);
  });
});

describe('firstLoadText', () => {
  it('ダウンロードを伴ったと分かっていれば秒を出す', () => {
    expect(firstLoadText(model('a', 'A', null, { loadFirstS: 132.54, firstLoadDownloaded: true }))).toBe('132.5');
  });

  it('キャッシュからだったなら秒を出さない', () => {
    expect(firstLoadText(model('a', 'A', null, { loadFirstS: 2.8, firstLoadDownloaded: false }))).toBe(
      CACHED_FIRST_LOAD,
    );
  });

  it('ダウンロードの有無が不明（古い JSON）なら「キャッシュ済み」と決めつけず、秒をそのまま出す', () => {
    expect(firstLoadText(model('a', 'A', null, { loadFirstS: 98.4, firstLoadDownloaded: null }))).toBe('98.4');
  });
});

describe('breakdownGroups', () => {
  it('1 モデルを 1 つのまとまり（tbody）にし、各データセットの行と総合の行を持たせる', () => {
    const results = fixtureResults();
    const groups = breakdownGroups(results);
    expect(groups.map((group) => group.model.key)).toEqual(results.models.map((item) => item.key));
    for (const group of groups) {
      expect(group.rows.map((row) => row.cells[0]?.text)).toEqual([
        ...results.datasets.map((dataset) => dataset.label),
        '総合',
      ]);
      expect(group.rows.map((row) => row.isOverall)).toEqual([false, false, false, true]);
    }
  });

  it('失敗のあるデータセットの失敗セルだけ注意として示す', () => {
    const qat = breakdownGroups(fixtureResults()).find((group) => group.model.key === 'e2b-qat');
    const failedCells = qat?.rows.map((row) => row.cells[2]);
    expect(failedCells?.some((cell) => cell?.warning === true)).toBe(true);
    expect(failedCells?.every((cell) => cell?.warning === (cell?.text !== '0'))).toBe(true);
  });
});

const withOutliers = (outlierIds: readonly string[] | null): BenchmarkResults => ({
  ...fixtureResults(),
  outlierIds,
});

describe('runOutlierSentence', () => {
  it('run 全体の外れ値クリップ数を 1 回だけ言う', () => {
    expect(runOutlierSentence(withOutliers(['reazon-1', 'reazon-2', 'cv-3']))).toBe(
      'この実行で外れ値として記録されたクリップは計 3 件です（ID は集計 JSON の outlier_ids）。',
    );
  });

  it('outlier_ids が無い古い JSON では何も言わない', () => {
    expect(runOutlierSentence(withOutliers(null))).toBe('');
  });
});
