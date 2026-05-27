from dataclasses import dataclass, field

import cv2
import librosa
import numpy as np


@dataclass
class HighlightSegment:
    start_time: float
    end_time: float
    audio_score: float = 0.0
    visual_score: float = 0.0
    combined_score: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class RuleEngineConfig:
    segment_window: float = 3.0
    top_k: int = 5
    min_segment_gap: float = 1.0
    audio_weight: float = 0.5
    visual_weight: float = 0.5
    min_duration: float = 1.0
    max_duration: float = 15.0


class AudioAnalyzer:
    def __init__(self, sr: int = 22050):
        self.sr = sr

    def analyze(self, y: np.ndarray) -> np.ndarray:
        if len(y) == 0:
            return np.array([])

        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        if rms.max() > 0:
            rms = rms / rms.max()

        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        if onset_env.max() > 0:
            onset_env = onset_env / onset_env.max()

        min_len = min(len(rms), len(onset_env))
        rms = rms[:min_len]
        onset_env = onset_env[:min_len]

        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
        zcr = zcr[:min_len]
        if zcr.max() > 0:
            zcr = zcr / zcr.max()

        score = 0.4 * rms + 0.35 * onset_env + 0.25 * zcr
        return score

    def detect_silence(self, y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
        intervals = librosa.effects.split(y, top_db=top_db)
        return intervals


class VisualAnalyzer:
    def __init__(self):
        pass

    def analyze(self, frame_paths: list[str], fps: float) -> np.ndarray:
        if len(frame_paths) < 2:
            return np.array([])

        scores = []
        prev_gray = None

        for path in frame_paths:
            frame = cv2.imread(path)
            if frame is None:
                scores.append(0.0)
                continue

            if frame.ndim == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.ndim == 2:
                gray = frame
            else:
                scores.append(0.0)
                continue

            frame_score = 0.0

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion = float(np.mean(diff)) / 255.0
                frame_score += 0.5 * motion

            brightness = float(np.mean(gray)) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            frame_score += 0.3 * brightness_score

            contrast = float(np.std(gray)) / 128.0
            frame_score += 0.2 * min(contrast, 1.0)

            scores.append(frame_score)
            prev_gray = gray

        result = np.array(scores)
        if result.max() > 0:
            result = result / result.max()
        return result


class RuleEngine:
    def __init__(self, config: RuleEngineConfig | None = None):
        self.config = config or RuleEngineConfig()
        self.audio_analyzer = AudioAnalyzer()
        self.visual_analyzer = VisualAnalyzer()

    def detect(
        self,
        audio: np.ndarray,
        sr: int,
        frame_paths: list[str],
        fps: float,
        duration: float,
    ) -> list[HighlightSegment]:
        audio_scores = self.audio_analyzer.analyze(audio)
        visual_scores = self.visual_analyzer.analyze(frame_paths, fps)

        combined = self._align_and_combine(audio_scores, visual_scores, duration)
        segments = self._extract_segments(combined, duration)
        segments = self._merge_overlapping(segments)
        segments = self._filter_by_duration(segments)
        segments.sort(key=lambda s: s.combined_score, reverse=True)

        return segments[: self.config.top_k]

    def _align_and_combine(
        self, audio_scores: np.ndarray, visual_scores: np.ndarray, duration: float
    ) -> np.ndarray:
        target_len = max(
            int(duration / (self.config.segment_window / 2)),
            max(len(audio_scores), len(visual_scores)),
        )

        if len(audio_scores) > 0:
            audio_resampled = np.interp(
                np.linspace(0, len(audio_scores) - 1, target_len),
                np.arange(len(audio_scores)),
                audio_scores,
            )
        else:
            audio_resampled = np.zeros(target_len)

        if len(visual_scores) > 0:
            visual_resampled = np.interp(
                np.linspace(0, len(visual_scores) - 1, target_len),
                np.arange(len(visual_scores)),
                visual_scores,
            )
        else:
            visual_resampled = np.zeros(target_len)

        # 无音频时提高视觉权重，避免全零分数导致无输出
        audio_w = self.config.audio_weight if len(audio_scores) > 0 else 0.0
        visual_w = self.config.visual_weight if len(visual_scores) > 0 else 0.0
        total_w = audio_w + visual_w
        if total_w == 0:
            return np.zeros(target_len)
        return (audio_w * audio_resampled + visual_w * visual_resampled) / total_w

    def _extract_segments(
        self, scores: np.ndarray, duration: float
    ) -> list[HighlightSegment]:
        if len(scores) == 0:
            return []

        step = duration / len(scores)
        threshold = np.percentile(scores, 70)

        # 全零分数时不提取任何片段
        if threshold <= 0:
            return []

        segments: list[HighlightSegment] = []
        in_segment = False
        seg_start = 0.0

        for i, score in enumerate(scores):
            t = i * step
            if score >= threshold and not in_segment:
                seg_start = t
                in_segment = True
            elif score < threshold and in_segment:
                segments.append(
                    HighlightSegment(
                        start_time=seg_start,
                        end_time=t,
                        combined_score=float(np.mean(scores[int(seg_start / step) : i + 1])),
                    )
                )
                in_segment = False

        if in_segment:
            segments.append(
                HighlightSegment(
                    start_time=seg_start,
                    end_time=duration,
                    combined_score=float(np.mean(scores[int(seg_start / step) :])),
                )
            )

        return segments

    def _merge_overlapping(self, segments: list[HighlightSegment]) -> list[HighlightSegment]:
        if not segments:
            return segments

        segments.sort(key=lambda s: s.start_time)
        merged: list[HighlightSegment] = []
        current = segments[0]

        for seg in segments[1:]:
            if seg.start_time - current.end_time <= self.config.min_segment_gap:
                current.end_time = max(current.end_time, seg.end_time)
                current.combined_score = max(current.combined_score, seg.combined_score)
            else:
                merged.append(current)
                current = seg

        merged.append(current)
        return merged

    def _filter_by_duration(self, segments: list[HighlightSegment]) -> list[HighlightSegment]:
        return [
            s
            for s in segments
            if self.config.min_duration <= s.duration <= self.config.max_duration
        ]
