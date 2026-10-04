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

## 依存の固定

- モデル: `src/worker.ts` の `MODEL_REVISION`（Hugging Face のコミット SHA）。更新するときはこの値を変えるコミットとしてレビューする
  - 既知の制約: Transformers.js 4.3.0 は tokenizer ファイルの存在確認（`get_tokenizer_files`）に revision を渡さないため、この確認だけは `main` を参照する。中身のダウンロードは固定した revision から行われる
- ONNX Runtime の WASM: Transformers.js が依存する onnxruntime-web の dist をビルド時に `dist/ort/<版>/` へコピーし、同一オリジンから配信する（jsDelivr には取りに行かない）。Transformers.js はこの WASM を URL をキーに Cache API へ永続化するため、版をパスに含めて更新時に古いものが使われないようにしている
