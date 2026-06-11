"""VoiceEngine: PersonaPlex frame loop with an injection hook.

One step() per 80ms frame. Injection mechanism per docs/decisions/0001:
audio briefings (TTS, distinct voice) on the user channel, gated on output silence.

G2 invariant: this class returns model PCM; it never plays audio itself.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import rustymimi
import sentencepiece
import sphn
from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model as load_tts_model

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

FRAME = 1920
SAMPLE_RATE = 24000
KOKORO_MODEL = "prince-canuma/Kokoro-82M"

# Injection gating (docs/decisions/0001): pending briefing PCM starts draining
# only after the model's own output has been quiet for 2s (25 frames), with an
# 8s failsafe so a briefing can never starve forever.
#
# Why 25 and not the originally-planned 4 (320ms): controlled A/B at seed
# 42424242 (same question, same briefing, only this constant varied) showed the
# model treats a briefing landing 320ms after its own speech as barge-in
# ("Thanks, I heard it's going to be sunny and warm" — hallucination, zero fact
# uptake), while a ~2s gap reproduces the spike-C5 known-good result verbatim
# ("Got it, 31 Celsius and clear skies"). The spike's 6s fixed beat was also
# ~2s of effective post-quiet gap, so this matches the proven recipe.
INJECT_AFTER_QUIET_FRAMES = 25
INJECT_MAX_WAIT_FRAMES = 100
OUTPUT_QUIET_RMS = 0.01


class VoiceEngine:
    def __init__(
        self,
        system_prompt: str,
        voice: str = "NATF2",
        quantize_bits: int = 8,
        seed: int = -1,
        briefing_voice: str = "am_michael",
    ) -> None:
        seed_all(seed)
        hf_repo = DEFAULT_HF_REPO
        lm_config = get_lm_config(None, hf_repo)
        tokenizer_file = get_or_download_tokenizer(hf_repo, None)
        model_file, _ = get_or_download_model_file(hf_repo, quantize_bits, None)
        mimi_file = get_or_download_mimi(hf_repo, None)

        self._text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)
        load_lm_weights(model, lm_config, model_file, quantize_bits)

        self._gen = models.LmGen(
            model=model,
            max_steps=100000,
            text_sampler=utils.Sampler(temp=0.7, top_k=25),
            audio_sampler=utils.Sampler(temp=0.8, top_k=250),
            check=False,
            audio_silence_frame_cnt=int(0.5 * 12.5),
        )
        voice_dir = get_voice_prompt_dir(None, hf_repo)
        self._gen.load_voice_prompt_embeddings(resolve_voice_prompt(voice, None, voice_dir))
        self._gen.text_prompt_tokens = self._text_tokenizer.encode(wrap_with_system_tags(system_prompt))
        self._gen.reset_streaming()
        self._gen.step_system_prompts()

        self._mimi = rustymimi.Tokenizer(mimi_file, num_codebooks=8)

        self._briefing_voice = briefing_voice
        # Kokoro is lazy-loaded on first inject(): it is only needed once a
        # briefing arrives, and keeping it out of __init__ keeps engine
        # construction (and tests that never inject) cheaper.
        self._kokoro = None

        # Injection state
        self._pending_audio: np.ndarray | None = None
        self._draining = False
        self._quiet_frames = 0
        self._inject_waited = 0
        self.last_gate_wait_frames: int | None = None

    @property
    def pending_injection(self) -> bool:
        return self._pending_audio is not None

    def inject(self, briefing: str) -> None:
        """TTS the briefing (Kokoro, distinct voice) and queue it as user-channel audio."""
        if self._kokoro is None:
            self._kokoro = load_tts_model(model_path=KOKORO_MODEL)
        # Kokoro consumes the global mx/python RNG streams; snapshot and restore
        # them so mid-session synthesis never perturbs the seeded generation
        # trajectory (in the spikes the briefing was synthesized outside the
        # session — this keeps the engine mechanics equivalent). Seeding inside
        # the snapshot makes the briefing audio byte-identical for a given text
        # regardless of when inject() is called.
        mx_state = mx.random.state[0]
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            seed_all(0)
            with tempfile.TemporaryDirectory() as tmp_dir:
                prefix = Path(tmp_dir) / "briefing"
                generate_audio(
                    text=briefing,
                    model=self._kokoro,
                    voice=self._briefing_voice,
                    file_prefix=str(prefix),
                    audio_format="wav",
                    join_audio=True,
                    play=False,
                    verbose=False,
                )
                # Normalize to 24kHz mono float32 (Kokoro emits 24kHz, but sphn
                # resampling makes that an invariant rather than an assumption).
                pcm, _ = sphn.read(str(prefix.with_suffix(".wav")), sample_rate=SAMPLE_RATE)
        finally:
            mx.random.state[0] = mx_state
            random.setstate(py_state)
            np.random.set_state(np_state)
        self.inject_audio(pcm[0].astype(np.float32))

    def inject_audio(self, pcm_24k: np.ndarray) -> None:
        """Queue pre-rendered briefing PCM (the primitive inject() builds on)."""
        pcm = np.asarray(pcm_24k, dtype=np.float32).reshape(-1)
        if self._pending_audio is None:
            self._pending_audio = pcm
        else:
            self._pending_audio = np.concatenate([self._pending_audio, pcm])

    def step(self, mic_frame: np.ndarray | None) -> tuple[np.ndarray | None, str]:
        """One 80ms step. Returns (model PCM frame or None, text piece or '').
        Injection precedence (when gate is open): pending briefing PCM REPLACES the mic frame."""
        if self._pending_audio is not None and not self._draining:
            if (
                self._quiet_frames >= INJECT_AFTER_QUIET_FRAMES
                or self._inject_waited >= INJECT_MAX_WAIT_FRAMES
            ):
                self._draining = True
                self.last_gate_wait_frames = self._inject_waited
            else:
                self._inject_waited += 1

        if self._draining:
            # Once draining starts it runs to completion (never re-gated);
            # briefing PCM replaces the mic frame for its whole duration.
            chunk = self._pending_audio[:FRAME]
            remainder = self._pending_audio[FRAME:]
            if remainder.shape[-1] == 0:
                self._pending_audio = None
                self._draining = False
                self._inject_waited = 0
            else:
                self._pending_audio = remainder
            input_tokens = self._encode_pcm(chunk)
        elif mic_frame is not None:
            input_tokens = self._encode_pcm(np.asarray(mic_frame, dtype=np.float32))
        else:
            input_tokens = self._gen._encode_sine_frame()

        out_text_token = self._gen.step(input_tokens=input_tokens)
        text = ""
        if out_text_token is not None:
            tid = int(out_text_token[0].item())
            if tid not in (0, 1, 2, 3):
                text = self._text_tokenizer.id_to_piece(tid).replace("\u2581", " ")

        audio_out: np.ndarray | None = None
        audio_tokens = self._gen.last_audio_tokens()
        if audio_tokens is not None:
            decode = np.array(audio_tokens[:, :, None]).astype(np.uint32)
            audio_out = np.asarray(self._mimi.decode_step(decode))[0, 0]

        quiet = audio_out is None or float(np.sqrt(np.mean(np.square(audio_out)))) < OUTPUT_QUIET_RMS
        self._quiet_frames = self._quiet_frames + 1 if quiet else 0

        return audio_out, text

    def _encode_pcm(self, pcm: np.ndarray) -> mx.array:
        if pcm.shape[-1] < FRAME:
            pcm = np.pad(pcm, (0, FRAME - pcm.shape[-1]))
        encoded = self._mimi.encode_step(pcm[None, None, :].astype(np.float32))
        tokens = mx.array(encoded).transpose(0, 2, 1)[:, :, : self._gen.user_codebooks]
        if tokens.shape[1] != self._gen.user_codebooks:
            tokens = tokens.transpose(0, 2, 1)
        return tokens

    def reset_session(self) -> None:
        """End-of-session: clear pending injection + gating state; re-prime the model."""
        self._pending_audio = None
        self._draining = False
        self._quiet_frames = 0
        self._inject_waited = 0
        self.last_gate_wait_frames = None
        self._gen.reset_streaming()
        self._gen.step_system_prompts()
