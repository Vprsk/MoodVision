"""Модуль распознавания эмоций по голосу и детекции акустических пиков (Сцены 1, 2, 3)."""

import threading
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    pipeline,
)

from app.core.logging import get_logger
from contracts.schemas import AudioSegmentResult, Emotion

logger = get_logger("audio_branch")

DEFAULT_AUDIO_MODEL_ID = "superb/wav2vec2-base-superb-er"

_AUDIO_PIPELINE_CACHE: dict[tuple[str, int, str], Any] = {}
_AUDIO_PIPELINE_LOCK = threading.Lock()


def _build_audio_pipeline(
    model_id: str,
    device: int,
    dtype: torch.dtype,
    *,
    offline: bool,
) -> Any:
    load_kwargs = {"local_files_only": offline}
    hf_model = AutoModelForAudioClassification.from_pretrained(
        model_id, dtype=dtype, **load_kwargs
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, **load_kwargs)
    return pipeline(
        "audio-classification",
        model=hf_model,
        feature_extractor=feature_extractor,
        device=device,
        dtype=dtype,
    )


def _get_audio_pipeline(model_id: str, device: int, dtype: torch.dtype) -> Any:
    """Возвращает кэшированный на уровне процесса пайплайн классификации эмоций речи."""
    key = (model_id, int(device), str(dtype))
    with _AUDIO_PIPELINE_LOCK:
        cached = _AUDIO_PIPELINE_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            clf = _build_audio_pipeline(model_id, device, dtype, offline=True)
            logger.info("Аудио-эмоции (HF): %s — из локального кэша", model_id)
        except Exception:
            clf = _build_audio_pipeline(model_id, device, dtype, offline=False)
            logger.info("Аудио-эмоции (HF): %s — первая загрузка с HF Hub", model_id)
        _AUDIO_PIPELINE_CACHE[key] = clf
        return clf


class AudioAffectAnalyzer:
    """Анализатор эмоций голоса с предварительным фильтром тишины (VAD + Peak Gate)."""

    LABEL_MAP: dict[str, Emotion] = {
        "ang": Emotion.ANGRY,
        "angry": Emotion.ANGRY,
        "anger": Emotion.ANGRY,
        "sad": Emotion.SAD,
        "sadness": Emotion.SAD,
        "hap": Emotion.HAPPY,
        "happy": Emotion.HAPPY,
        "happiness": Emotion.HAPPY,
        "fea": Emotion.FEAR,
        "fear": Emotion.FEAR,
        "dis": Emotion.DISGUST,
        "disgust": Emotion.DISGUST,
        "sur": Emotion.SURPRISE,
        "surprise": Emotion.SURPRISE,
        "neu": Emotion.NEUTRAL,
        "neutral": Emotion.NEUTRAL,
    }

    def __init__(
        self,
        model_id: str = DEFAULT_AUDIO_MODEL_ID,
        device: int = -1,
        torch_dtype: torch.dtype = torch.float32,
        sample_rate: int = 16000,
        vad_rms_threshold: float = 0.015,
        loud_spike_threshold: float = 0.18,
        audio_conf_threshold: float = 0.35,
        lazy_load: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.sample_rate = sample_rate
        self.vad_rms_threshold = vad_rms_threshold
        self.loud_spike_threshold = loud_spike_threshold
        self.audio_conf_threshold = audio_conf_threshold
        self._classifier: Any | None = None
        if not lazy_load:
            self._classifier = _get_audio_pipeline(
                self.model_id, self.device, self.torch_dtype
            )

    def _ensure_classifier(self) -> Any:
        if self._classifier is None:
            self._classifier = _get_audio_pipeline(
                self.model_id, self.device, self.torch_dtype
            )
        return self._classifier

    def analyze(self, audio_chunk: np.ndarray | None = None) -> AudioSegmentResult:
        """Оценивает аудиофрагмент (моно, float32 в [-1, 1]). Безопасен при audio_chunk=None."""
        if audio_chunk is None or audio_chunk.size == 0:
            return AudioSegmentResult(speech_detected=False)

        audio_f32 = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(audio_f32))))
        peak = float(np.max(np.abs(audio_f32)))

        speech_detected = rms >= self.vad_rms_threshold
        is_spike = peak >= self.loud_spike_threshold

        # В тишине нейросеть не вызывается (0 мс нагрузка на GPU)
        if not speech_detected and not is_spike:
            return AudioSegmentResult(
                speech_detected=False,
                rms_energy=round(rms, 4),
                is_loud_spike=False,
                audio_emotion=Emotion.UNKNOWN,
                audio_confidence=0.0,
            )

        try:
            clf = self._ensure_classifier()
            preds = clf(
                {"raw": audio_f32, "sampling_rate": self.sample_rate},
                top_k=1,
            )
            if preds:
                raw_label = str(preds[0]["label"]).lower()
                score = float(preds[0]["score"])
                mapped_em = self.LABEL_MAP.get(raw_label, Emotion.UNKNOWN)
                if score < self.audio_conf_threshold:
                    mapped_em = Emotion.UNKNOWN
                return AudioSegmentResult(
                    speech_detected=speech_detected,
                    rms_energy=round(rms, 4),
                    is_loud_spike=is_spike,
                    audio_emotion=mapped_em,
                    audio_confidence=round(score, 4),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ошибка классификации аудио: %s", exc)

        # Фоллбек по акустическому пику, если модель недоступна
        fallback_em = Emotion.ANGRY if is_spike else Emotion.UNKNOWN
        fallback_conf = 0.75 if is_spike else 0.0
        return AudioSegmentResult(
            speech_detected=speech_detected,
            rms_energy=round(rms, 4),
            is_loud_spike=is_spike,
            audio_emotion=fallback_em,
            audio_confidence=fallback_conf,
        )