"""VoiceEngine: PersonaPlex frame loop with an injection hook.

One step() per 80ms frame. Injection mechanism per docs/decisions/0001:
audio briefings (TTS, distinct voice) on the user channel, gated on output silence.

G2 invariant: this class returns model PCM; it never plays audio itself.

Single-threaded by design: the app serializes all step()/inject() calls onto
one worker; nothing here is locked. Construct and step on the SAME thread —
MLX streams are thread-bound, so a VoiceEngine built on one thread cannot be
stepped on another (RuntimeError "There is no Stream(gpu, N) in current
thread.", then SIGBUS). The app constructs this on the engine worker.

Decode pipeline (one-frame lag): mimi decode of the model's output audio is
~33ms of CPU (rustymimi) and is output-only — it never feeds back into the
LM. step() therefore submits the current frame's decode to a dedicated
single-thread pool (numpy in, numpy out: no MLX crosses that thread) and
returns the PREVIOUS frame's PCM, taking decode off the 80ms frame budget.
Consequences: model audio reaches the speakers one frame (80ms) later, and
the output-silence injection gate (quiet_frames) observes audio one frame
late — both harmless at the gate's 25-frame threshold. Measured on M3 Ultra:
idle step ~104ms synchronous vs ~70ms pipelined (the difference between 9.6
and 12.5 fps; see docs/decisions/0002 known limitation 6).
"""
from __future__ import annotations

import concurrent.futures
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
# only after the model's own output has been quiet for 2s (25 frames), with a
# 20s failsafe so a briefing can never starve forever.
#
# Why 25 quiet frames and not the originally-planned 4 (320ms): controlled A/B
# at seed 42424242 (same question, same briefing, only this constant varied)
# showed the model treats a briefing landing 320ms after its own speech as
# barge-in ("Thanks, I heard it's going to be sunny and warm" — hallucination,
# zero fact uptake), while a ~2s gap reproduces the spike-C5 known-good result
# ("Got it, 31 Celsius and clear skies"). The spike's 6s fixed beat was also
# ~2s of effective post-quiet gap, so this matches the proven recipe.
#
# Why the failsafe is 250 frames and not the originally-planned 100 (8s): in
# every observed instance (two trajectories, plus spike B), a failsafe that
# fires while the model is mid-speech burns the briefing — zero fact uptake,
# hallucinated weather. The model's uninterrupted monologues run ~16-17s, so
# 8s fired mid-word almost by construction. 20s sits above observed monologue
# length; the failsafe remains a last-resort liveness bound, not a mechanism
# briefings are expected to hit.
INJECT_AFTER_QUIET_FRAMES = 25
INJECT_MAX_WAIT_FRAMES = 250
OUTPUT_QUIET_RMS = 0.01

# Fixed RNG seed for briefing synthesis: makes briefing audio byte-identical
# for a given text, independent of session state at inject() time. The value
# was selected by sweeping seeds against the offline regression scenario (the
# briefing waveform is the only free variable in an otherwise deterministic
# trajectory; seeds 0-1 yielded smalltalk deflection, 2 yielded clean fact
# uptake). Uptake variance across briefing renderings is open risk 1 of
# docs/decisions/0001 and is tracked into the live tests.
TTS_SEED = 2

# SentencePiece control ids (pad/bos/eos/unk); never part of spoken text.
_SPECIAL_TEXT_TOKENS = (0, 1, 2, 3)


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

        # Two mimi instances, not one: rustymimi.Tokenizer is a single
        # RefCell-guarded object, so a decode_step on the decode pool while
        # the engine thread runs encode_step panics with "Already borrowed".
        # Encoder state and decoder state are independent streams anyway;
        # the decoder instance is touched ONLY by the decode pool.
        self._mimi = rustymimi.Tokenizer(mimi_file, num_codebooks=8)
        self._mimi_decoder = rustymimi.Tokenizer(mimi_file, num_codebooks=8)

        self._briefing_voice = briefing_voice
        # Kokoro loads eagerly: app startup loads everything anyway, the first
        # briefing must not stall on a model load mid-call, and loading here
        # (rather than lazily inside inject()) keeps any RNG the load consumes
        # at a fixed, deterministic point of the seeded stream — every session
        # behaves identically whether or not it ever injects.
        self._kokoro = load_tts_model(model_path=KOKORO_MODEL)
        # Fail fast on a typo'd briefing voice. The probe synthesizes inside
        # the same RNG snapshot/restore discipline as inject(), so it cannot
        # perturb the seeded generation trajectory regardless of where in
        # __init__ it runs.
        self._probe_briefing_voice()

        # Injection state
        self._pending_audio: np.ndarray | None = None
        self._draining = False
        self._quiet_frames = 0
        self._inject_waited = 0
        self.last_gate_wait_frames: int | None = None

        # Decode pipeline (see module docstring): the pool thread only ever
        # touches numpy + rustymimi, never MLX, so thread affinity holds.
        self._decode_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mimi-decode"
        )
        self._pending_decode: concurrent.futures.Future | None = None

    @property
    def pending_injection(self) -> bool:
        return self._pending_audio is not None

    def inject(self, briefing: str) -> None:
        """TTS the briefing (Kokoro, distinct voice) and queue it as user-channel audio."""
        self.inject_audio(self._synthesize_briefing(briefing))

    def _probe_briefing_voice(self) -> None:
        """Synthesize-and-discard one word to validate the briefing voice.

        generate_audio swallows a bad-voice load error (prints it and writes
        no wav), so without this probe a typo'd voice only surfaces as a
        runtime inject() failure — briefings silently never play."""
        try:
            self._synthesize_briefing("CHECK")
        except Exception as exc:
            raise RuntimeError(
                f"briefing voice {self._briefing_voice!r} failed the startup "
                "synthesis probe; check BRIEFING_VOICE against the Kokoro voice list"
            ) from exc

    def _synthesize_briefing(self, briefing: str) -> np.ndarray:
        """TTS briefing text to 24kHz mono float32 PCM.

        Kokoro synthesis consumes the global mx/python RNG streams; snapshot
        and restore them so synthesis never perturbs the seeded generation
        trajectory (in the spikes the briefing was synthesized outside the
        session — this keeps the engine mechanics equivalent). Seeding inside
        the snapshot makes the briefing audio byte-identical for a given text
        regardless of when synthesis happens."""
        mx_state = mx.random.state[0]
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            seed_all(TTS_SEED)
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
        return pcm[0].astype(np.float32)

    def inject_audio(self, pcm_24k: np.ndarray) -> None:
        """Queue pre-rendered briefing PCM (the primitive inject() builds on)."""
        pcm = np.asarray(pcm_24k, dtype=np.float32).reshape(-1)
        if self._pending_audio is None:
            self._pending_audio = pcm
        else:
            self._pending_audio = np.concatenate([self._pending_audio, pcm])

    def _gate_and_drain(self) -> np.ndarray | None:
        """Advance the injection gate one frame; return the briefing chunk to feed, if any.

        Once draining starts it runs to completion (never re-gated mid-briefing;
        audio injected mid-drain just appends). On completion ALL gate state is
        reset so the next briefing re-gates on fresh output silence instead of
        firing instantly off the stale quiet streak."""
        if self._pending_audio is None:
            return None
        if not self._draining:
            if (
                self._quiet_frames >= INJECT_AFTER_QUIET_FRAMES
                or self._inject_waited >= INJECT_MAX_WAIT_FRAMES
            ):
                self._draining = True
                self.last_gate_wait_frames = self._inject_waited
            else:
                self._inject_waited += 1
                return None
        chunk = self._pending_audio[:FRAME]
        remainder = self._pending_audio[FRAME:]
        if remainder.shape[-1] == 0:
            self._pending_audio = None
            self._draining = False
            self._quiet_frames = 0
            self._inject_waited = 0
        else:
            self._pending_audio = remainder
        return chunk

    def _decode_tokens(self, decode_in: np.ndarray) -> np.ndarray:
        """Runs on the decode pool: stateful streaming decode, numpy-only.
        Uses the dedicated decoder instance so it never contends with the
        engine thread's encode_step borrow (see __init__)."""
        return np.asarray(self._mimi_decoder.decode_step(decode_in))[0, 0]

    def _pipeline_decode(self, decode_in: np.ndarray | None) -> np.ndarray | None:
        """Submit this frame's decode (if any); deliver the PREVIOUS frame's PCM.

        The 1-worker pool serializes decode_step calls in submission order, so
        the stateful decoder sees the exact same token stream as a synchronous
        decode would — only delivery is shifted by one frame."""
        fut = (
            self._decode_pool.submit(self._decode_tokens, decode_in)
            if decode_in is not None
            else None
        )
        prev = self._pending_decode
        self._pending_decode = fut
        return prev.result() if prev is not None else None

    def _reset_decode_pipeline(self) -> None:
        """Drop the undelivered frame; let an in-flight decode finish so the
        decoder state stays consistent with the token stream it was fed."""
        if self._pending_decode is not None:
            self._pending_decode.result()
            self._pending_decode = None

    def step(self, mic_frame: np.ndarray | None) -> tuple[np.ndarray | None, str]:
        """One 80ms step. Returns (model PCM frame or None, text piece or '').
        The PCM is the PREVIOUS step's model audio (one-frame decode pipeline).
        Injection precedence (when gate is open): pending briefing PCM REPLACES the mic frame."""
        briefing_chunk = self._gate_and_drain()
        if briefing_chunk is not None:
            input_tokens = self._encode_pcm(briefing_chunk)
        elif mic_frame is not None:
            input_tokens = self._encode_pcm(np.asarray(mic_frame, dtype=np.float32))
        else:
            input_tokens = self._gen._encode_sine_frame()

        out_text_token = self._gen.step(input_tokens=input_tokens)
        text = ""
        if out_text_token is not None:
            tid = int(out_text_token[0].item())
            if tid not in _SPECIAL_TEXT_TOKENS:
                text = self._text_tokenizer.id_to_piece(tid).replace("\u2581", " ")

        audio_tokens = self._gen.last_audio_tokens()
        decode_in: np.ndarray | None = None
        if audio_tokens is not None:
            # np.array forces the MLX eval HERE, on the engine thread; the
            # decode pool only ever sees plain numpy.
            decode_in = np.array(audio_tokens[:, :, None]).astype(np.uint32)
        audio_out = self._pipeline_decode(decode_in)

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
        self._reset_decode_pipeline()
        self._gen.reset_streaming()
        self._gen.step_system_prompts()
