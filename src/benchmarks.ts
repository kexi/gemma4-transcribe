import { createTooltip, renderBarChart, type BarChartSpec, type BarDatum } from './benchmarks/chart.ts';
import { EXCLUDED_NOTE } from './benchmarks/compare.ts';
import { formatCount, formatFixed, formatPercent, MISSING, seriesClass, shortCommit } from './benchmarks/format.ts';
import { buildHeadline } from './benchmarks/headline.ts';
import { niceScale } from './benchmarks/scale.ts';
import { parseResults, type BenchmarkResults, type ModelResult } from './benchmarks/schema.ts';
import {
  breakdownGroups,
  breakdownHeaders,
  CACHED_FIRST_LOAD,
  groupOf,
  OVERALL_LABEL,
  summaryTable,
  runOutlierSentence,
  type Cell,
  type CellKind,
} from './benchmarks/table.ts';

/**
 * GitHub Pages のサブパス（/<repo>/）でも動くよう、ページ自身からの相対 URL で取る。
 * 本番ビルドが eval/results/*.json を dist/benchmarks/ にコピーする（tsdown.config.ts）
 */
const RESULTS_URL = './benchmarks/latest.json';
const REPOSITORY_URL = 'https://github.com/kexi/gemma4-transcribe';

/**
 * データセットの性質の説明（数値ではないので JSON に無ければここから補う）。
 * JSON 側に description があればそちらを優先する
 */
const DATASET_DESCRIPTIONS: Readonly<Record<string, string>> = {
  jsut: '1 人の話者による朗読。静かな録音で、最も易しい条件',
  reazon: 'テレビ放送の音声。雑音・BGM・話者の重なりがあり、最も難しい条件',
  cv: '一般の協力者による読み上げ。話者・マイク・録音環境がさまざま',
};

/** environment の既知の項目名を日本語の見出しにする。未知の項目は JSON の名前のまま出す。 */
const ENVIRONMENT_LABELS: Readonly<Record<string, string>> = {
  chrome: 'Chrome',
  platform: 'OS',
  gpu: 'GPU（WebGPU アダプタ）',
  machine: 'マシン',
};

function mustGet<T extends HTMLElement>(id: string, type: new () => T): T {
  const found = document.getElementById(id);
  if (!(found instanceof type)) throw new Error(`#${id} が見つかりません`);
  return found;
}

const headline = mustGet('bench-headline', HTMLParagraphElement);
const errorBox = mustGet('bench-error', HTMLDivElement);
const content = mustGet('bench-content', HTMLDivElement);

/** 要素を作って textContent を入れる。JSON 由来の文字列を HTML として解釈させないため innerHTML は使わない。 */
function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  options: { className?: string; text?: string } = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (options.className !== undefined) node.className = options.className;
  if (options.text !== undefined) node.textContent = options.text;
  node.append(...children);
  return node;
}

function showError(title: string, details: readonly string[]): void {
  headline.textContent = '';
  headline.hidden = true;
  const list = element('ul');
  for (const detail of details) list.append(element('li', { text: detail }));
  errorBox.replaceChildren(element('strong', { text: title }), list);
  errorBox.hidden = false;
}

function showEmpty(): void {
  headline.textContent = '';
  headline.hidden = true;
  const message = element(
    'div',
    { className: 'bench-empty' },
    element('strong', { text: 'まだ結果がありません' }),
    element('p', {
      text: '評価ハーネス（just eval）を実行して eval/results/latest.json をコミットすると、ここに精度と速度の比較が表示されます。',
    }),
  );
  content.replaceChildren(message);
  content.hidden = false;
}

const toPercentUnits = (fraction: number | null): number | null => (fraction === null ? null : fraction * 100);

function chartSpecs(results: BenchmarkResults): BarChartSpec[] {
  const groups: { key: string | null; title: string; subtitle: string }[] = [
    ...results.datasets.map((dataset) => ({
      key: dataset.key,
      title: dataset.label,
      subtitle: dataset.n === null ? '' : `${formatCount(dataset.n)} クリップ`,
    })),
    { key: null, title: OVERALL_LABEL, subtitle: '全データセットのマイクロ平均' },
  ];
  const specs = groups.map(({ key, title, subtitle }) => ({
    title,
    subtitle,
    bars: results.models.map((model): BarDatum => {
      const metrics = groupOf(model, key);
      return {
        modelKey: model.key,
        modelLabel: model.label,
        value: toPercentUnits(metrics?.cerNorm ?? null),
        n: metrics?.n ?? null,
        failed: metrics?.failed ?? null,
        median: metrics?.cerNormMedian ?? null,
        outliers: metrics?.outliers ?? null,
      };
    }),
  }));
  // 全グラフで同じ軸にする（グラフをまたいで棒の長さを比べられるように）
  const maxValue = Math.max(0, ...specs.flatMap((spec) => spec.bars.map((bar) => bar.value ?? 0)));
  const scale = niceScale(maxValue);
  const layoutBars = specs.flatMap((spec) => spec.bars);
  return specs.map((spec) => ({ ...spec, scale, layoutBars }));
}

function renderLegend(results: BenchmarkResults): HTMLElement {
  const legend = element('ul', { className: 'bench-legend' });
  legend.setAttribute('aria-label', '凡例');
  for (const model of results.models) {
    const swatch = element('span', { className: `bench-swatch ${seriesClass(model.key)}` });
    swatch.setAttribute('aria-hidden', 'true');
    legend.append(element('li', {}, swatch, model.label));
  }
  return legend;
}

function renderCharts(results: BenchmarkResults): HTMLElement {
  const specs = chartSpecs(results);
  const tooltip = createTooltip();
  const grid = element('div', { className: 'bench-multiples' });
  const cells = specs.map((spec) => {
    const plot = element('div', { className: 'bench-chart-plot' });
    const figure = element(
      'figure',
      { className: 'bench-chart' },
      element(
        'figcaption',
        {},
        element('span', { className: 'bench-chart-title', text: spec.title }),
        element('span', { className: 'bench-chart-subtitle', text: spec.subtitle }),
      ),
      plot,
    );
    grid.append(figure);
    return { spec, plot };
  });

  // 実際の描画幅で描き直す（viewBox で縮めると文字まで小さくなり、狭い画面で読めなくなるため）
  const font = `12px ${getComputedStyle(document.body).fontFamily}`;
  const lastWidths = new Map<HTMLElement, number>();
  const draw = (): void => {
    for (const { spec, plot } of cells) {
      const width = Math.floor(plot.clientWidth);
      const isUnchanged = lastWidths.get(plot) === width;
      if (isUnchanged || width === 0) continue;
      lastWidths.set(plot, width);
      plot.replaceChildren(renderBarChart(spec, width, font, tooltip));
    }
  };
  new ResizeObserver(() => draw()).observe(grid);
  requestAnimationFrame(draw);

  return element(
    'section',
    { className: 'bench-section' },
    element('h2', { text: 'データセット別 CER（正規化）' }),
    element('p', {
      className: 'bench-note',
      text: '短いほど誤りが少ない。全グラフで軸を共有しています。棒にポインタを合わせる（またはフォーカスする）とクリップ数・中央値・外れ値を表示します。',
    }),
    // 1 系列なら色で区別するものが無く見出しが系列名を兼ねるので、凡例は 2 モデル以上のときだけ出す
    ...(results.models.length >= 2 ? [renderLegend(results)] : []),
    grid,
  );
}

/** 表のセル 1 つ。最良は太字と読み上げ用の「（最良）」、注意は色と文字で示す。 */
function renderCell(cell: Cell): HTMLTableCellElement {
  const td = element('td');
  if (cell.best) {
    const strong = element('strong', { className: 'bench-best', text: cell.text });
    strong.title = '最良';
    td.append(strong, element('span', { className: 'bench-visually-hidden', text: '（最良）' }));
  } else {
    td.textContent = cell.text;
  }
  if (cell.warning) td.classList.add('bench-warning');
  if (cell.kind === 'text') td.classList.add('bench-text-cell');
  if (cell.kind === 'ids') td.classList.add('bench-ids');
  return td;
}

/** 行見出し（色の見本 + モデル名 + 計測状況）。 */
function modelHeader(model: ModelResult, status: string | null): HTMLTableCellElement {
  const swatch = element('span', { className: `bench-swatch ${seriesClass(model.key)}` });
  swatch.setAttribute('aria-hidden', 'true');
  const th = element('th', {}, swatch, model.label);
  if (status !== null) th.append(element('span', { className: 'bench-model-note', text: status }));
  return th;
}

function headerRow(headers: readonly { text: string; kind: CellKind }[]): HTMLTableRowElement {
  const row = element('tr');
  for (const header of headers) {
    const th = element('th', { text: header.text });
    th.setAttribute('scope', 'col');
    // 文字の列（モデル名・データセット名・ID）は見出しも左寄せにして、数値の列と揃え方を合わせる
    if (header.kind !== 'number') th.classList.add('bench-text-cell');
    row.append(th);
  }
  return row;
}

function renderSummaryTable(results: BenchmarkResults): HTMLElement {
  const summary = summaryTable(results);
  const table = element('table', { className: 'bench-table' });
  table.append(element('caption', { text: 'モデル別の CER・速度・失敗数' }));
  const headers = [
    { text: 'モデル', kind: 'text' as const },
    ...summary.headers.map((text) => ({ text, kind: 'number' as const })),
  ];
  table.append(element('thead', {}, headerRow(headers)));
  const body = element('tbody');
  for (const { model, status, cells } of summary.rows) {
    const th = modelHeader(model, status);
    th.setAttribute('scope', 'row');
    body.append(element('tr', {}, th, ...cells.map(renderCell)));
  }
  table.append(body);
  return element(
    'section',
    { className: 'bench-section' },
    element('h2', { text: '精度と速度' }),
    element('p', {
      className: 'bench-note',
      text: `CER はすべて % で、小さいほど良い。太字は各列の最良値で、全クリップの計測を終えたモデルの間だけで比べます（途中のモデルには「${EXCLUDED_NOTE}」と添えます）。初回の読み込み秒はキャッシュの有無で意味が変わるため比べず、キャッシュから読んだときは「${CACHED_FIRST_LOAD}」と表示します。失敗したクリップは CER と RTF から除き、失敗数として別に示します。「${MISSING}」はその値が計測されていないことを表します。`,
    }),
    element('div', { className: 'bench-table-scroll' }, table),
  );
}

/** 外れ値と中央値の表。行 = モデル × データセット（+ 総合）で、1 モデルを 1 つの tbody にまとめる。 */
function renderBreakdown(results: BenchmarkResults): HTMLElement {
  const table = element('table', { className: 'bench-table' });
  table.append(element('caption', { text: 'モデル・データセット別の内訳' }));
  table.append(element('thead', {}, headerRow(breakdownHeaders(results))));
  for (const { model, rows } of breakdownGroups(results)) {
    // scope=rowgroup の見出しが指す範囲は tbody 単位なので、モデルごとに tbody を分ける
    // Why not 全モデルを 1 つの tbody に入れる: 支援技術が 1 つ目のモデル名を全モデルの行の見出しとして読んでしまうため
    const body = element('tbody');
    rows.forEach(({ isOverall, cells }, index) => {
      const row = element('tr');
      const isFirstOfModel = index === 0;
      if (isFirstOfModel) {
        const th = modelHeader(model, null);
        th.setAttribute('scope', 'rowgroup');
        th.rowSpan = rows.length;
        row.append(th);
      }
      row.append(...cells.map(renderCell));
      if (isOverall) row.classList.add('bench-overall-row');
      body.append(row);
    });
    table.append(body);
  }
  const outlierText = formatPercent(results.outlierThreshold, 0);
  return element(
    'section',
    { className: 'bench-section' },
    element('h2', { text: '外れ値と中央値' }),
    element('p', {
      className: 'bench-note',
      text: `CER はクリップをまたいだマイクロ平均なので、短いクリップで同じ文を繰り返すような大きな誤りや、参照テキストが音声の一部しか書き起こしていないクリップ（クリップ単位の CER が ${outlierText} 超）に引っ張られます。中央値と、外れ値を除いた CER（参考値）で、その影響の大きさを確かめられます。参考値は順位付けには使いません。${runOutlierSentence(results)}`,
    }),
    element('div', { className: 'bench-table-scroll' }, table),
  );
}

function definitionList(entries: readonly (readonly [string, Node | string])[]): HTMLElement {
  const list = element('dl', { className: 'bench-dl' });
  for (const [term, description] of entries) list.append(element('dt', { text: term }), element('dd', {}, description));
  return list;
}

function link(href: string, text: string): HTMLAnchorElement {
  const anchor = element('a', { text });
  anchor.href = href;
  return anchor;
}

/** Hugging Face のリポジトリ名・revision として妥当なときだけリンクにする（JSON の文字列を URL に直接埋め込まないため）。 */
function datasetLink(repo: string | null, revision: string | null): Node | string {
  if (repo === null) return MISSING;
  const isRepoName = /^[\w.-]+\/[\w.-]+$/.test(repo);
  if (!isRepoName) return repo;
  const isRevision = revision !== null && /^[0-9a-f]{7,40}$/i.test(revision);
  const href = `https://huggingface.co/datasets/${repo}${isRevision ? `/tree/${revision}` : ''}`;
  return link(href, isRevision ? `${repo} @ ${shortCommit(revision)}` : repo);
}

function renderMethod(results: BenchmarkResults): HTMLElement {
  const outlierText = formatPercent(results.outlierThreshold, 0);
  const metrics = definitionList([
    ['CER 厳密', 'NFC と空白の除去だけ。全角半角・句読点・漢字とかなの違いもすべて誤りとして数える。'],
    [
      'CER 正規化',
      'NFKC・句読点と記号の除去・英字の小文字化の後に比べる。一般的な日本語 ASR の評価に近く、このページの主指標。',
    ],
    [
      'CER 読み',
      '正規化の後に形態素解析（fugashi + unidic-lite）で読み（カタカナ）にして比べる。漢字の選び方や送り仮名の違いを無視した「音の正しさ」。',
    ],
    [
      '集計方法',
      `CER は総編集数 / 参照の総文字数（マイクロ平均）。中央値はクリップ単位の CER（正規化）の中央値。外れ値は、この実行のどれか 1 つのモデルでクリップ単位の CER（正規化）が ${outlierText} を超えたクリップ。外れ値を除いた CER（参考値）では、全モデルから同じクリップを除く（モデルごとに違うクリップ集合で比べないため）。`,
    ],
    ['RTF', '推論時間の合計 / 音声長の合計。1 未満なら実時間より速い。'],
    [
      '読み込み秒',
      `初回はその計測で最初に読み込んだ時間（ダウンロードを含む）。計測前からモデルがブラウザのキャッシュにあったときは「${CACHED_FIRST_LOAD}」と表示する（ダウンロードの有無が記録されていない古い結果では、その時間をそのまま表示する）。キャッシュはダウンロード済みのモデルをブラウザのキャッシュから読み込んだ時間。ウォームアップは読み込み後に 1 回だけ計測せずに流した文字起こしの時間（WebGPU のシェーダのコンパイル等。RTF には含めない）。`,
    ],
  ]);

  const datasetRows = results.datasets.map((dataset): readonly [string, Node | string] => {
    const description = dataset.description ?? DATASET_DESCRIPTIONS[dataset.key] ?? '';
    const facts = [
      dataset.n === null ? '' : `${formatCount(dataset.n)} クリップ`,
      dataset.totalAudioS === null ? '' : `音声 ${formatFixed(dataset.totalAudioS / 60, 1)} 分`,
      dataset.excludedOver30s === null ? '' : `30 秒超で除外 ${formatCount(dataset.excludedOver30s)} 件`,
    ].filter((fact) => fact !== '');
    const body = element('span');
    if (description !== '') body.append(`${description}。`);
    if (facts.length > 0) body.append(`${facts.join('、')}。`, ' ');
    body.append(datasetLink(dataset.repo, dataset.revision));
    return [dataset.label, body];
  });

  const notes = element('ul', { className: 'bench-notes' });
  for (const note of results.notes) notes.append(element('li', { text: note }));

  return element(
    'section',
    { className: 'bench-section' },
    element('h2', { text: '指標とデータ' }),
    metrics,
    element('h3', { text: 'データセット' }),
    element('p', {
      className: 'bench-note',
      text: '各データセットから決まった手順で選んだクリップを使います（30 秒を超える音声は除外）。表記ゆれ（全角半角・句読点・漢字とかな）は CER の 3 種類で扱いを変えて示します。音声・参照テキスト・推論結果は再配布しないため、このページには数値と ID だけを載せています。',
    }),
    definitionList(datasetRows),
    ...(results.notes.length > 0 ? [element('h3', { text: '補足' }), notes] : []),
  );
}

function formatTimestamp(value: string | null): string {
  if (value === null) return MISSING;
  const date = new Date(value);
  const isValidDate = !Number.isNaN(date.getTime());
  if (!isValidDate) return value;
  return `${date.toLocaleString('ja-JP', { timeZone: 'UTC', dateStyle: 'medium', timeStyle: 'short' })} UTC`;
}

function renderEnvironment(results: BenchmarkResults): HTMLElement {
  const commit = results.gitCommit;
  const isSha = commit !== null && /^[0-9a-f]{7,40}$/i.test(commit);
  const commitNode = element('span');
  if (commit === null) commitNode.append(MISSING);
  else commitNode.append(isSha ? link(`${REPOSITORY_URL}/commit/${commit}`, shortCommit(commit)) : commit);
  // 未コミットの変更を含むコードで測った結果は、その commit を見ても完全には再現できないことを明示する
  if (results.gitDirty) commitNode.append('（未コミットの変更あり）');
  const modelRows = results.models.map((model): readonly [string, Node | string] => {
    const isRepoName = model.id !== null && /^[\w.-]+\/[\w.-]+$/.test(model.id);
    const isRevision = model.revision !== null && /^[0-9a-f]{7,40}$/i.test(model.revision);
    const isLinkable = isRepoName && isRevision;
    if (!isLinkable) return [model.label, model.id ?? MISSING];
    return [
      model.label,
      link(
        `https://huggingface.co/${model.id}/tree/${model.revision}`,
        `${model.id} @ ${shortCommit(model.revision ?? '')}`,
      ),
    ];
  });
  return element(
    'section',
    { className: 'bench-section' },
    element('h2', { text: '実行環境' }),
    definitionList([
      ['実行 ID', results.runId],
      ['集計日時', formatTimestamp(results.generatedAt)],
      ['git commit', commitNode],
      ...results.environment.map(([name, value]) => [ENVIRONMENT_LABELS[name] ?? name, value] as const),
    ]),
    element('h3', { text: 'モデル（revision 固定）' }),
    definitionList(modelRows),
  );
}

function render(results: BenchmarkResults): void {
  headline.textContent = buildHeadline(results);
  headline.hidden = false;
  content.replaceChildren(
    renderCharts(results),
    renderSummaryTable(results),
    renderBreakdown(results),
    renderMethod(results),
    renderEnvironment(results),
  );
  content.hidden = false;
}

async function main(): Promise<void> {
  let response: Response;
  try {
    // 結果は同じ URL のまま更新されるので、古いキャッシュを見せないよう毎回検証させる
    response = await fetch(RESULTS_URL, { cache: 'no-cache' });
  } catch (error) {
    showError('結果を取得できませんでした', [String(error)]);
    return;
  }
  const isNotPublished = response.status === 404;
  if (isNotPublished) {
    showEmpty();
    return;
  }
  if (!response.ok) {
    showError('結果を取得できませんでした', [`HTTP ${response.status} ${response.statusText}`]);
    return;
  }
  let raw: unknown;
  try {
    raw = await response.json();
  } catch (error) {
    showError('結果の JSON が壊れています', [String(error)]);
    return;
  }
  const parsed = parseResults(raw);
  if (!parsed.ok) {
    showError('結果の JSON がこのページの想定と違います', parsed.errors);
    return;
  }
  render(parsed.data);
}

void main();
