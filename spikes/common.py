"""Shared spike harness: load PersonaPlex once, step PCM frames, collect output."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
import rustymimi
import sentencepiece
import sphn

from personaplex_mlx import models, utils
from personaplex_mlx.persona_utils import (
    DEFAULT_HF_REPO,
    get_lm_config,
    get_or_download_mimi,
    get_or_download_model_file,
    get_or_download_tokenizer,
    get_voice_prompt_dir,
    load_lm_weights,
    resolve_voice_prompt,
    seed_all,
    wrap_with_system_tags,
)

SAMPLE_RATE = 24000
FRAME = 1920  # 80ms @ 24kHz, one model step
OUT = Path(__file__).parent / "out"  # all spike I/O is __file__-relative: cwd-independent

SPIKE_SYSTEM_PROMPT = (
    "You are Moneypenny, a sharp and warm personal assistant. "
    "You sometimes receive short system briefings through your earpiece; "
    "they are facts gathered for you, not something the user said. "
    "When you receive a briefing, weave its facts naturally into your reply "
    "in your own words. Never mention briefings, systems, or tools. "
    "Never guess facts about weather, schedules, or the home; "
    "if you have not been briefed, say you will check."
)


@dataclass
class SpikeSession:
    gen: models.LmGen
    text_tokenizer: sentencepiece.SentencePieceProcessor
    audio_tokenizer: "rustymimi.Tokenizer"
    out_pcm: list = field(default_factory=list)
    out_text: list = field(default_factory=list)

    def step_pcm(self, pcm_frame: np.ndarray, text_token: int | None = None) -> None:
        """Step one 1920-sample frame of user audio (pads short frames)."""
        if pcm_frame.shape[-1] < FRAME:
            pcm_frame = np.pad(pcm_frame, (0, FRAME - pcm_frame.shape[-1]))
        encoded = self.audio_tokenizer.encode_step(pcm_frame[None, None, :].astype(np.float32))
        tokens = mx.array(encoded).transpose(0, 2, 1)[:, :, : self.gen.user_codebooks]
        if tokens.shape[1] != self.gen.user_codebooks:
            tokens = tokens.transpose(0, 2, 1)
        self._step(input_tokens=tokens, text_token=text_token)

    def step_sine(self, text_token: int | None = None, mute_assistant: bool = False) -> None:
        """Step one frame of synthetic 'user silence' (sine tokens).

        mute_assistant=True additionally forces the assistant's audio stream to
        silence tokens — exactly what step_system_prompts() does while feeding
        the system prompt (generate.py:319-332). Spike A tests both variants.
        """
        kwargs = {"input_tokens": self.gen._encode_sine_frame(), "text_token": text_token}
        if mute_assistant:
            kwargs["moshi_tokens"] = self.gen._encode_zero_frame()
        self._step(**kwargs)

    def _step(self, **kwargs) -> None:
        forced = kwargs.get("text_token") is not None
        out_text_token = self.gen.step(**kwargs)
        # When we force a text token, LmGen.step still returns a *sampled* token
        # (generate.py:208-245) — a phantom that was never part of the model's
        # spoken output. Keep the transcript clean: skip capture on forced steps.
        if out_text_token is not None and not forced:
            tid = int(out_text_token[0].item())
            if tid not in (0, 1, 2, 3):
                piece = self.text_tokenizer.id_to_piece(tid)
                self.out_text.append(piece.replace("\u2581", " "))
        audio_tokens = self.gen.last_audio_tokens()
        if audio_tokens is not None:
            decode = np.array(audio_tokens[:, :, None]).astype(np.uint32)
            self.out_pcm.append(self.audio_tokenizer.decode_step(decode))

    def step_wav(self, path: str | Path) -> None:
        pcm, _ = sphn.read(str(path), sample_rate=SAMPLE_RATE)
        total = pcm.shape[-1]
        for i in range(0, total, FRAME):
            self.step_pcm(pcm[0, i : i + FRAME])

    def run_free(self, seconds: float) -> None:
        """Let the model talk: feed sine 'silence' frames."""
        for _ in range(int(seconds * SAMPLE_RATE / FRAME)):
            self.step_sine()

    def save(self, wav_name: str, text_name: str) -> None:
        """Writes into spikes/out/ regardless of cwd."""
        pcm = np.concatenate(self.out_pcm, axis=-1)
        wav_path, text_path = OUT / wav_name, OUT / text_name
        rustymimi.write_wav(str(wav_path), pcm[0, 0], sample_rate=SAMPLE_RATE)
        text_path.write_text(json.dumps("".join(self.out_text), ensure_ascii=False))
        print(f"wrote {wav_path}\ntranscript: {''.join(self.out_text)!r}")


def load_session(system_prompt: str = SPIKE_SYSTEM_PROMPT, seed: int = 42424242) -> SpikeSession:
    seed_all(seed)
    hf_repo = DEFAULT_HF_REPO
    lm_config = get_lm_config(None, hf_repo)
    tokenizer_file = get_or_download_tokenizer(hf_repo, None)
    model_file, _ = get_or_download_model_file(hf_repo, 8, None)  # 8-bit per spec
    mimi_file = get_or_download_mimi(hf_repo, None)

    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    load_lm_weights(model, lm_config, model_file, 8)

    gen = models.LmGen(
        model=model,
        max_steps=100000,
        text_sampler=utils.Sampler(temp=0.7, top_k=25),
        audio_sampler=utils.Sampler(temp=0.8, top_k=250),
        check=False,
        audio_silence_frame_cnt=int(0.5 * 12.5),
    )
    voice_dir = get_voice_prompt_dir(None, hf_repo)
    gen.load_voice_prompt_embeddings(resolve_voice_prompt("NATF2", None, voice_dir))
    gen.text_prompt_tokens = text_tokenizer.encode(wrap_with_system_tags(system_prompt))
    gen.reset_streaming()
    gen.step_system_prompts()

    audio_tokenizer = rustymimi.Tokenizer(mimi_file, num_codebooks=8)
    return SpikeSession(gen=gen, text_tokenizer=text_tokenizer, audio_tokenizer=audio_tokenizer)
