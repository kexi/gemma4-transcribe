export interface TranscriptionLanguage {
  /** プロンプトに埋め込む英語の言語名。 */
  name: string;
  /** UI に表示する名前。 */
  label: string;
}

/** Gemma 4 の ASR が対応する言語のうち、UI で選べるもの。 */
export const LANGUAGES: readonly TranscriptionLanguage[] = [
  { name: 'Japanese', label: '日本語' },
  { name: 'English', label: 'English' },
  { name: 'Chinese', label: '中文' },
  { name: 'Korean', label: '한국어' },
  { name: 'Spanish', label: 'Español' },
  { name: 'French', label: 'Français' },
  { name: 'German', label: 'Deutsch' },
];

/** モデルカードが示す ASR 用プロンプトを、指定言語で組み立てる。 */
export function buildTranscriptionPrompt(language: string): string {
  return [
    `Transcribe the following speech segment in ${language} into ${language} text.`,
    '',
    'Follow these specific instructions for formatting the answer:',
    '* Only output the transcription, with no newlines.',
    '* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.',
  ].join('\n');
}
