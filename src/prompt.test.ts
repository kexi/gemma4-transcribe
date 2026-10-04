import { describe, expect, it } from 'vitest';

import { buildTranscriptionPrompt, LANGUAGES } from './prompt.ts';

describe('buildTranscriptionPrompt', () => {
  it('モデルカードの ASR プロンプトに言語名を埋め込む', () => {
    const prompt = buildTranscriptionPrompt('Japanese');
    expect(prompt.split('\n')[0]).toBe('Transcribe the following speech segment in Japanese into Japanese text.');
    expect(prompt).toContain('* Only output the transcription, with no newlines.');
  });
});

describe('LANGUAGES', () => {
  it('先頭（既定の選択肢）が日本語である', () => {
    expect(LANGUAGES[0]?.name).toBe('Japanese');
  });
});
