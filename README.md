# gemma4-transcribe

Gemma 4 E2B をブラウザの WebGPU で動かして、音声を文字起こしする静的サイトです。音声はブラウザの外に送られません。

**デモ**: https://kexi.github.io/gemma4-transcribe/

- モデル: [onnx-community/gemma-4-E2B-it-ONNX](https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX)（q4f16、初回ダウンロード約 3.4GB。以降はブラウザのキャッシュから読み込む）
- 推論: [Transformers.js](https://huggingface.co/docs/transformers.js) + WebGPU（Web Worker 内で実行）
- 入力: マイク録音、または音声・動画ファイル（ブラウザがデコードできる形式）
- 動作環境: WebGPU が使えるデスクトップ版 Chrome / Edge

## LiteRT.js（LiteRT-LM）を使っていない理由

[`@litert-lm/core`](https://www.npmjs.com/package/@litert-lm/core) は 0.17.1 時点で text-in / text-out のみのプレビューで、
Web 向けの `gemma-4-E2B-it-web.litertlm` も「Currently the model is text-only」とされていて音声エンコーダを含まない。
JS の `Engine` API にも音声バックエンドを指定する設定が無いため、音声入力は Transformers.js で扱っている。

## 仕組み

- Gemma 4 の音声入力は 1 回あたり最大 30 秒（750 トークン × 40ms）。超えた分は processor が黙って切り捨てるため、30 秒ごとの区間に分けて順に推論する
- 区間の境界は固定長なので、単語の途中で切れることがある
- 0.1 秒未満の末尾は捨てる（音声トークンを作れず、推論が失敗するため）
- プロンプトはモデルカードの ASR 用テンプレートをそのまま使う

## 開発

```sh
direnv allow        # または nix develop
just install
just serve          # http://localhost:8000
just check          # CI と同じゲート
```

`main` への push で CI（`just check`）が通ると GitHub Pages にデプロイされます。
新しく fork / 作り直したリポジトリでは、最初に一度だけ Pages の公開元を GitHub Actions にしておく必要があります。

```sh
gh api -X POST repos/<owner>/gemma4-transcribe/pages -f build_type=workflow
```

## 評価

公開データセットの音声を、アプリと同じ Worker・同じ推論経路で実際の Chrome（WebGPU）に文字起こしさせ、精度と速度を測る。
モデル（E2B / E2B QAT / E4B / E4B QAT）を同じ物差しで比べるためのもので、評価ページ自体は本番の `dist/` には含まれない。
Python 側（`eval/`）は uv のプロジェクトで、devshell に uv と Python 3.13 が入っている。`just install` が `eval/uv.lock` どおりに venv を作り（`--locked`）、
`just check` の中の `eval-test` はその venv をオフラインで使う（ロックの再計算や PyPI への問い合わせをしない）。

```sh
just eval e2b,e4b 100     # prepare → build → run → score をまとめて実行（この recipe は位置引数のみ）
# 個別に実行する場合
just eval-prepare 100     # 各データセットから 100 件を選び eval/data/ に wav と manifest.json を書く
just eval-build           # 評価ページを dist-eval/ にビルド
just eval-run e2b,e4b 10  # Chrome を起動して先頭 10 件を評価（limit を省くと全件）
just eval-run models=e2b,e2b-qat,e4b,e4b-qat   # 4 モデルを 1 つの run_id で順に測る（name=value でも書ける）
just eval-score           # 最新の run の全モデルを eval/results/<run_id>.{md,json} と latest.json に集計
```

- データセット（revision は `eval/gemma4_eval/prepare.py` で固定）
  - jsut: [japanese-asr/ja_asr.jsut_basic5000](https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000)（朗読・1 話者）
  - reazon: [japanese-asr/ja_asr.reazonspeech_test](https://huggingface.co/datasets/japanese-asr/ja_asr.reazonspeech_test)（放送・雑音あり）
  - cv: [japanese-asr/ja_asr.common_voice_8_0](https://huggingface.co/datasets/japanese-asr/ja_asr.common_voice_8_0)（一般話者）
- サンプリング: 行番号 i を `sha256("<key>:<i>")` の昇順に並べ、30 秒を超える音声（Gemma の 1 区間の上限）を飛ばしながら先頭 N 件を取る。件数を増やしても既に選ばれたクリップは変わらない。飛ばした件数は manifest の `excluded_over_30s` に残る
- `eval-run` は Playwright でインストール済みの Google Chrome を画面付きで起動する（`eval/.chrome-profile/` を使い回すので、モデルのダウンロードは初回だけ）。モデルのキャッシュはオリジン単位なので、サーバは固定ポート（8765）で立てる。塞がっていれば 8766〜8774 を順に試し、全部塞がっているときだけ空きポートを使う（このときはモデルを再ダウンロードする）
- 1 モデルごとに次の順で測る
  1. クリップ 0 件のページでモデルを読み込ませ、ダウンロードを済ませる（このときの読み込み時間が「初回 load」）。
     読み込みの直前と直後に Cache API にあるそのモデル（id + revision）のファイル数を数え、増えていれば「初回 DL あり」（`first_load_downloaded: true`）、
     変わらなければ「なし」（キャッシュから読んだ）、数えられなければ「不明」（`null`）として記録する。初回 load がダウンロードを含むのは「あり」のときだけ
  2. 計測用のページを開き直してキャッシュから読み込む（「load」。表の読み込み時間はこちら）
  3. 先頭クリップを 1 回だけ計測せずに文字起こしする（「warmup」。WebGPU のシェーダのコンパイルや AudioContext の初期化を計測から外す）。ページを開き直すたびに行う
  4. クリップを順に文字起こしして測る
- 結果は 1 クリップごとに `eval/results/raw/<run_id>/<model>.jsonl` へ追記する。途中で止まったら `just eval-run <同じモデル> run=<run_id>` で記録済みのクリップを飛ばして続きから測れる。run_id は `YYYYMMDDTHHMMSSZ` の形式で、`run.json` がある既存の run だけを指定できる（打ち間違いが新しい run になって `latest.json` を奪わないよう、形式違いや存在しない run_id は止まる）。再開しても `run.json` の `chrome` は最初の実行の値のまま残し、再開時の Chrome は `resumes` の各要素に記録する。offset / limit / 言語は最初の実行の `run.json` の値を使い、違う値を明示すると止まる（別条件の結果を 1 つの run に混ぜないため）。manifest を作り直した場合も止まる
- モデルの読み込みは 40 分、1 クリップ（ウォームアップを含む）は 5 分でタイムアウトし、タイムアウトしたクリップは失敗として記録して次に進む。ページがクラッシュしたときは処理中だったクリップから開き直し、同じクリップで 2 回クラッシュしたら失敗として記録する

### データの扱い

元コーパスの利用条件（JSUT: 研究・個人利用、ReazonSpeech: 著作権法 30 条の 4 の情報解析目的、Common Voice: CC0 だが話者の特定は禁止）に従い、
音声・参照テキスト・推論結果のテキストは再配布しない。`eval/data/` と `eval/results/raw/` は `.gitignore` 済みで、
コミットするのは数値・ID・実行環境だけの `eval/results/<run_id>.md`、`eval/results/<run_id>.json`、`eval/results/latest.json` に限る。
JSON はベンチマークページが読む集計で（形は `eval/gemma4_eval/score.py` の `build_page_json`、版は `schema`）、
実行環境として Chrome の版、WebGPU アダプタの情報（vendor / architecture / description）、OS と機種、git commit と未コミットの変更の有無を含む。
`latest.json` はこれまでに集計した中で最も新しい run を指し、古い run を集計し直しても巻き戻らない（run_id の形式に合わないファイルやディレクトリは比べない）。

### 指標

- CER は文字単位の編集距離のマイクロ平均（総編集数 / 参照総文字数）。正規化を変えて 3 種類出す
  - strict: NFC と空白の除去だけ。句読点・全角半角・漢字とかなの違いもすべて誤り
  - norm: NFKC、句読点と記号（Unicode カテゴリ P\*, S\*）の除去、英字の小文字化。一般的な日本語 ASR 評価に近い
  - reading: norm の後に fugashi + unidic-lite で読み（カタカナ）に変換して比べる。漢字の選び方や送り仮名の違いを無視した「音の正しさ」。
    英数字は全角大文字にしてから解析するので、「NHK」は辞書の読み「エヌエイチケー」になる（読みの無い語は norm と同じ表記のまま）
- RTF は推論時間の合計 / 音声長の合計（1 未満なら実時間より速い）。デコード込みの値も併記する。ウォームアップの 1 回は含めない
- 読み込み時間は、キャッシュ済みのモデルの読み込み（load）と、その run で最初の読み込み（初回 load）を分けて出す。初回 load がダウンロードを含むのは、初回 DL（`first_load_downloaded`）が `true` のときだけ
- 失敗したクリップは CER と RTF から除き、失敗数として必ず表に出す
- 外れ値: ReazonSpeech は参照テキストが音声の一部しか書き起こしていないクリップがあり、正しく聞き取ってもクリップ単体の CER が 200〜300% になる。
  見出しの CER はこうしたクリップも除かない全件のマイクロ平均のままにし（何を外れ値とみなすかで結果を動かせないように）、
  感度分析として、データセット × モデルごとにクリップ別 norm CER の中央値、除いた外れ値の件数、外れ値を除いたマイクロ平均（参考値）を別に出す。
  外れ値はモデルごとではなく run 全体でクリップ単位に決める（どれか 1 つのモデルで norm CER が 100% を超えたクリップ）。同じクリップをすべてのモデルから除き、
  モデルごとに違うクリップ集合で比べないようにする。外れ値のクリップ ID と件数は集計 JSON の `outlier_ids` / `outlier_count` と Markdown に 1 回だけ載せる

## 依存の固定

- モデル: `src/worker.ts` の `MODEL_REVISION`（Hugging Face のコミット SHA）。更新するときはこの値を変えるコミットとしてレビューする
  - 既知の制約: Transformers.js 4.3.0 は tokenizer ファイルの存在確認（`get_tokenizer_files`）に revision を渡さないため、この確認だけは `main` を参照する。中身のダウンロードは固定した revision から行われる
- ONNX Runtime の WASM: Transformers.js が依存する onnxruntime-web の dist をビルド時に `dist/ort/<版>/` へコピーし、同一オリジンから配信する（jsDelivr には取りに行かない）。Transformers.js はこの WASM を URL をキーに Cache API へ永続化するため、版をパスに含めて更新時に古いものが使われないようにしている
