import type { DataType } from '@huggingface/transformers';

/** Transformers.js の from_pretrained に渡す dtype（全セッション共通か、セッション名ごとの対応表）。 */
export type ModelDtype = DataType | Readonly<Record<string, DataType>>;

export interface ModelOption {
  /** UI・localStorage・Worker 間で使う短い識別子。 */
  key: string;
  label: string;
  /** Hugging Face のリポジトリ ID。 */
  id: string;
  /**
   * リポジトリのコミット SHA。
   * main を追うと第三者のリポジトリ更新がそのまま全訪問者に届くため、更新はこの値を変えるコミットとしてレビューする
   */
  revision: string;
  dtype: ModelDtype;
  /** 初回ダウンロード量の目安（GB）。UI の表示に使う。 */
  approxGigabytes: number;
}

/**
 * QAT mobile 版はセッションごとに量子化が違う（画像エンコーダだけ fp16）。
 * リポジトリの config.json の transformers.js_config と同じ内容だが、revision と一緒にこちらで固定してレビュー対象にする
 */
const QAT_MOBILE_DTYPE: Readonly<Record<string, DataType>> = {
  audio_encoder: 'q2f16',
  decoder_model_merged: 'q2f16',
  embed_tokens: 'q2f16',
  vision_encoder: 'fp16',
};

/** 音声入力に対応した Gemma 4（E2B / E4B）の ONNX 版。先頭が既定。 */
export const MODELS: readonly ModelOption[] = [
  {
    key: 'e2b',
    label: 'Gemma 4 E2B（q4f16）',
    id: 'onnx-community/gemma-4-E2B-it-ONNX',
    revision: '9f4bef82ea6e296bc69f8a2f5939f73af81b07a6',
    // q4f16 が WebGPU で最小。q4 は fp32 演算になり VRAM と転送量が増えるため採らない
    dtype: 'q4f16',
    approxGigabytes: 3.4,
  },
  {
    key: 'e2b-qat',
    label: 'Gemma 4 E2B QAT mobile（q2f16）',
    id: 'onnx-community/gemma-4-E2B-it-qat-mobile-ONNX',
    revision: '5cd5514efd375abf2801c856a3936b259cc00133',
    dtype: QAT_MOBILE_DTYPE,
    approxGigabytes: 2.6,
  },
  {
    key: 'e4b',
    label: 'Gemma 4 E4B（q4f16）',
    id: 'onnx-community/gemma-4-E4B-it-ONNX',
    revision: '843f250f23bc91754def1e0f0db390dacd1e6b05',
    dtype: 'q4f16',
    approxGigabytes: 5.2,
  },
  {
    key: 'e4b-qat',
    label: 'Gemma 4 E4B QAT mobile（q2f16）',
    id: 'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX',
    revision: '4d18aa8b54e354bec4705e4a4894f5bbf8956c3d',
    dtype: QAT_MOBILE_DTYPE,
    approxGigabytes: 3.6,
  },
];

/** 不明なキー（古い localStorage の値など）は既定モデルにフォールバックする。 */
export function findModel(key: string | null | undefined): ModelOption {
  const found = MODELS.find((model) => model.key === key);
  if (found !== undefined) return found;
  const [fallback] = MODELS;
  if (fallback === undefined) throw new Error('MODELS が空です');
  return fallback;
}
