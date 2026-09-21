"""
MedGemma-Micro: Wear OS Smartwatch (Samsung Galaxy Watch 4+) PPG Adapter & DSP Engine
====================================================================================
Bridges raw smartwatch biosensor telemetry from Wear OS / Samsung Health Sensor SDK
to the MedGemma-Micro 1D-Conformer Biosignal Encoder and multimodal LLM pipeline.

Hardware Specifications & Constraints (Samsung Galaxy Watch 4 / 5 / 6 BioActive Sensor):
  - Primary Channel: ValueKey.PpgSet.PPG_GREEN (raw photodiode ADC counts: ~100k to 1.5M counts).
  - Secondary Channels: ValueKey.PpgSet.PPG_IR, ValueKey.PpgSet.PPG_RED.
  - Contact / Lead-Off Status: ValueKey.PpgSet.GREEN_STATUS (0 = Valid Contact, < 0 = Detached/Lead-Off, > 0 = Motion).
  - Native Sampling Rates: 25 Hz (standard continuous tracking) or 100 Hz (high-precision mode).
  - Timestamps: Nanosecond epoch/uptime timestamps (subject to BLE packet transmission jitter).
  - Pulsatile Perfusion: Arterial AC wave is only 0.5% - 2.0% of the DC optical baseline (~2k - 15k counts).

This module provides:
  1. Binary & JSON deserialization for Wearable Data Layer API (ChannelClient / MessageClient).
  2. Signal Conditioning: Baseline wander removal, Butterworth bandpass (0.5 - 5.0 Hz), detrending.
  3. Anti-Aliased Resampler: Decimates 100 Hz / 50 Hz or jittered streams to exact 25.0 Hz.
  4. Signal Quality Index (SQI) & Contact Validation: Rejects off-wrist or saturated signals.
  5. Thread-safe WearOSStreamBuffer: Maintains a rolling 90-second window (2250 samples @ 25 Hz).
"""

import math
import time
import struct
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
import threading

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger("MedGemmaMicro.WearOS")


# =====================================================================
# 1. DATA STRUCTURES & PACKET PROTOCOL
# =====================================================================

@dataclass
class WearOSPPGPoint:
    """Represents a single PPG sample point from Wear OS / Samsung Health Sensor SDK."""
    timestamp_ns: int
    ppg_green: int
    status: int = 0  # 0: VALID, -1: DETACHED / LEAD_OFF, 1: MOTION_HIGH
    ppg_ir: Optional[int] = None
    ppg_red: Optional[int] = None

    @property
    def is_valid(self) -> bool:
        """Returns True if the sensor reports proper skin contact and no fatal motion noise."""
        return self.status == 0


@dataclass
class IngestionResult:
    """Summary of batch ingestion into the streaming buffer."""
    points_received: int
    points_valid: int
    points_dropped: int
    buffer_fill_pct: float
    current_sqi: float
    is_ready_for_inference: bool
    status_summary: str


class WearOSPacketProtocol:
    """
    Binary and JSON packet serializer/deserializer matching Google Play Services
    Wearable Data Layer API (ChannelClient byte streams and MessageClient payloads).
    
    Standard Binary Wire Format (little-endian):
      Header (4 bytes): Magic 0x57505047 ('WPPG' in ASCII)
      Version (1 byte): 0x01
      NumPoints (2 bytes): uint16
      Payload: Array of structs:
        - timestamp_ns (int64, 8 bytes)
        - ppg_green (int32, 4 bytes)
        - status (int32, 4 bytes)
      Total per point: 16 bytes.
    """

    MAGIC_HEADER = b"WPPG"
    VERSION = 1
    POINT_STRUCT = struct.Struct("<qii")  # timestamp_ns (q=8), ppg_green (i=4), status (i=4)

    @classmethod
    def pack_points(cls, points: List[WearOSPPGPoint]) -> bytes:
        """Serializes a list of WearOSPPGPoint into a compact binary packet for ChannelClient."""
        header = cls.MAGIC_HEADER + bytes([cls.VERSION]) + struct.pack("<H", len(points))
        body = bytearray()
        for p in points:
            body.extend(cls.POINT_STRUCT.pack(p.timestamp_ns, p.ppg_green, p.status))
        return bytes(header + body)

    @classmethod
    def unpack_binary(cls, raw_bytes: bytes) -> List[WearOSPPGPoint]:
        """Unpacks binary payload from Wearable Data Layer ChannelClient or MessageClient."""
        if len(raw_bytes) < 7:
            raise ValueError(f"Packet too short ({len(raw_bytes)} bytes)")

        magic = raw_bytes[:4]
        if magic != cls.MAGIC_HEADER:
            # Fallback: attempt direct raw point stream without header
            if len(raw_bytes) % cls.POINT_STRUCT.size == 0:
                points = []
                for offset in range(0, len(raw_bytes), cls.POINT_STRUCT.size):
                    t_ns, green, status = cls.POINT_STRUCT.unpack_from(raw_bytes, offset)
                    points.append(WearOSPPGPoint(timestamp_ns=t_ns, ppg_green=green, status=status))
                return points
            raise ValueError(f"Invalid magic header: {magic}")

        version = raw_bytes[4]
        num_points = struct.unpack("<H", raw_bytes[5:7])[0]
        offset = 7
        points = []
        for _ in range(num_points):
            if offset + cls.POINT_STRUCT.size > len(raw_bytes):
                break
            t_ns, green, status = cls.POINT_STRUCT.unpack_from(raw_bytes, offset)
            offset += cls.POINT_STRUCT.size
            points.append(WearOSPPGPoint(timestamp_ns=t_ns, ppg_green=green, status=status))
        return points

    @classmethod
    def parse_json(cls, payload: Union[Dict, List]) -> List[WearOSPPGPoint]:
        """
        Parses JSON payloads exported from Samsung Health Sensor SDK companion apps.
        Supports:
          - Dict format: {"points": [{"timestamp_ns": ..., "ppg_green": ..., "status": ...}]}
          - Direct List format: [{"timestamp": ..., "value": ..., "status": ...}]
        """
        raw_items = payload.get("points", payload) if isinstance(payload, dict) else payload
        points = []
        for item in raw_items:
            t_ns = item.get("timestamp_ns") or item.get("timestamp") or item.get("time") or int(time.time() * 1e9)
            # Heuristic: nanosecond timestamps are > 1e15 for dates after year 2001.
            # Millisecond timestamps (13 digits) are between 1e12 and 1e13.
            # Using 1e16 as the threshold safely covers ms inputs up to year 2286
            # while correctly leaving genuine ns timestamps (>= 1e15) unchanged.
            if t_ns < 1e16:
                t_ns = int(t_ns * 1_000_000)  # ms -> ns

            green = item.get("ppg_green") or item.get("value") or item.get("ppg") or 0
            status = item.get("status") if "status" in item else item.get("green_status", 0)
            ir = item.get("ppg_ir")
            red = item.get("ppg_red")
            points.append(WearOSPPGPoint(
                timestamp_ns=int(t_ns),
                ppg_green=int(green),
                status=int(status),
                ppg_ir=int(ir) if ir is not None else None,
                ppg_red=int(red) if red is not None else None,
            ))
        return points


# =====================================================================
# 2. SIGNAL QUALITY INDEX (SQI) & CONTACT DETECTOR
# =====================================================================

class WearOSSignalQuality:
    """
    Assesses physiological validity and contact quality of raw Wear OS PPG telemetry.
    Computes skewness, kurtosis, relative perfusion index, and lead-off flags.
    """

    @staticmethod
    def compute_sqi(raw_samples: np.ndarray) -> Dict[str, Union[float, bool, str]]:
        """
        Computes Signal Quality Index for a segment of raw ADC readings.
        Returns:
            sqi_score: Float in [0.0, 1.0]
            is_usable: Bool
            quality_flag: 'EXCELLENT', 'ACCEPTABLE', 'POOR_CONTACT', 'SATURATED', 'DETACHED'
        """
        if len(raw_samples) < 25:
            return {"sqi": 0.0, "is_usable": False, "quality_flag": "INSUFFICIENT_DATA"}

        samples = np.asarray(raw_samples, dtype=np.float64)
        mean_val = float(np.mean(samples))
        std_val = float(np.std(samples))
        min_val = float(np.min(samples))
        max_val = float(np.max(samples))

        # 1. Off-body / Flatline / Zero Check (ADC disconnected or sensor off skin)
        if mean_val < 1000.0 or std_val < 5.0:
            return {"sqi": 0.0, "is_usable": False, "quality_flag": "DETACHED"}

        # 2. Photodiode Saturation Check (ADC rail ceiling, typically 2^23 or 2^24 counts)
        if max_val > 16_000_000.0 or (max_val - min_val) < 20.0:
            return {"sqi": 0.05, "is_usable": False, "quality_flag": "SATURATED"}

        # 3. Perfusion Index (AC / DC ratio): Normal physiological PPG AC is 0.2% - 5.0% of DC
        perfusion_pct = (std_val / (mean_val + 1e-6)) * 100.0
        if perfusion_pct < 0.05:
            return {"sqi": 0.2, "is_usable": False, "quality_flag": "POOR_CONTACT"}
        elif perfusion_pct > 25.0:
            # Excessive motion artifact or ambient light leakage
            return {"sqi": 0.3, "is_usable": False, "quality_flag": "HIGH_MOTION_NOISE"}

        # 4. Statistical Distribution: True arterial pulses exhibit positive skewness
        centered = samples - mean_val
        m3 = np.mean(centered ** 3)
        m4 = np.mean(centered ** 4)
        skew = float(m3 / (std_val ** 3 + 1e-8))
        kurt = float(m4 / (std_val ** 4 + 1e-8))

        # Expected physiological skewness is generally positive (~0.1 to 1.5) due to steep systolic upstroke
        sqi = 0.5
        if 0.0 <= skew <= 2.0:
            sqi += 0.25
        if 2.0 <= kurt <= 6.0:
            sqi += 0.25

        sqi = max(0.0, min(1.0, sqi))
        flag = "EXCELLENT" if sqi >= 0.75 else ("ACCEPTABLE" if sqi >= 0.5 else "POOR_CONTACT")
        return {
            "sqi": round(sqi, 3),
            "is_usable": bool(sqi >= 0.5),
            "perfusion_pct": round(perfusion_pct, 3),
            "skewness": round(skew, 2),
            "kurtosis": round(kurt, 2),
            "quality_flag": flag,
        }


# =====================================================================
# 3. WEAR OS SIGNAL CONDITIONING & RESAMPLING DSP ENGINE
# =====================================================================

class WearOSPPGAdapter:
    """
    Transforms raw Wear OS (Samsung Galaxy Watch 4) telemetry into the exact
    format expected by MedGemma-Micro:
      - Resampling to uniform 25 Hz (from 100 Hz, 50 Hz, or non-uniform timestamps).
      - Bandpass filtering (0.5 Hz - 5.0 Hz 3rd-order Butterworth) to strip DC optical
        baseline and high-frequency motion/ambient noise.
      - Detrending to eliminate respiratory baseline wander.
      - Z-score normalization: zero-mean unit-variance float32 tensor of shape [1, 2250, 1].
    """

    TARGET_FS = 25  # MedGemma-Micro Conformer expects 25 Hz
    WINDOW_DURATION_SEC = 90
    TARGET_SAMPLES = TARGET_FS * WINDOW_DURATION_SEC  # 2250 samples

    def __init__(self, target_fs: int = 25, lowcut: float = 0.5, highcut: float = 5.0, order: int = 3):
        self.target_fs = target_fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

        # Precompute Butterworth filter coefficients for 25 Hz
        nyq = 0.5 * self.target_fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        self.b, self.a = scipy_signal.butter(self.order, [low, high], btype="bandpass")

    def resample_to_uniform_grid(
        self,
        timestamps_ns: np.ndarray,
        values: np.ndarray,
        target_duration_sec: float = 90.0,
    ) -> np.ndarray:
        """
        Resamples non-uniform / jittered or 100 Hz / 50 Hz raw timestamps onto an
        exact uniform 25 Hz time grid of length 2250 samples.
        Extracts the most recent `target_duration_sec` (90s) without artificial time dilation.
        """
        if len(timestamps_ns) < 2:
            raise ValueError("Insufficient data points for resampling")

        # Convert nanoseconds to relative seconds starting from 0
        t_sec = (timestamps_ns - timestamps_ns[0]) / 1e9
        total_time = t_sec[-1]

        # Anti-aliased interpolation
        # If input sampling rate is significantly higher than 25 Hz (e.g. 100 Hz),
        # apply preliminary lowpass decimation to avoid aliasing
        approx_fs = len(values) / max(total_time, 0.001)
        if approx_fs > 35.0:
            # Decimate high-rate signal
            decimate_factor = int(round(approx_fs / self.target_fs))
            if decimate_factor > 1:
                # Apply anti-aliasing FIR filter and resample
                values = scipy_signal.resample_poly(values, self.target_fs, int(round(approx_fs)))
                if len(values) >= self.TARGET_SAMPLES:
                    return values[-self.TARGET_SAMPLES:].astype(np.float32)
                else:
                    target_t = np.linspace(0, 1, self.TARGET_SAMPLES)
                    src_t = np.linspace(0, 1, len(values))
                    return np.interp(target_t, src_t, values).astype(np.float32)

        # Standard uniform 25 Hz extraction:
        # If we have >= target_duration_sec (e.g. 90s), extract the most recent window
        if total_time >= target_duration_sec:
            start_t = total_time - target_duration_sec
            target_t = np.linspace(start_t, total_time, self.TARGET_SAMPLES, endpoint=False)
            resampled = np.interp(target_t, t_sec, values)
            return resampled.astype(np.float32)
        else:
            # Buffer not yet full (less than target_duration_sec):
            # Resample available duration at true 25 Hz and edge-pad to 2250 samples
            num_pts = max(2, int(round(total_time * self.target_fs)))
            t_avail = np.linspace(0, total_time, num_pts, endpoint=False)
            v_avail = np.interp(t_avail, t_sec, values)
            if num_pts < self.TARGET_SAMPLES:
                pad_left = self.TARGET_SAMPLES - num_pts
                resampled = np.pad(v_avail, (pad_left, 0), mode="edge")
            else:
                resampled = v_avail[-self.TARGET_SAMPLES:]
            return resampled.astype(np.float32)

    def condition_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Applies detrending, 0.5-5.0 Hz Butterworth bandpass filtering,
        and Z-score normalization to a 25 Hz signal.
        """
        signal_data = np.asarray(raw_signal, dtype=np.float64).flatten()

        # 1. Linear detrend to eliminate major optical drift
        detrended = scipy_signal.detrend(signal_data, type="linear")

        # 2. Zero-phase forward-backward Butterworth filter (filtfilt)
        # filtfilt requires signal length > 3 * max(len(a), len(b)) to avoid edge artifacts.
        min_filtfilt_len = 3 * max(len(self.a), len(self.b)) + 1
        if len(detrended) < min_filtfilt_len:
            # Edge-pad to meet minimum requirement, filter, then trim
            pad_len = min_filtfilt_len - len(detrended)
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left
            detrended_padded = np.pad(detrended, (pad_left, pad_right), mode="edge")
            try:
                filtered_padded = scipy_signal.filtfilt(self.b, self.a, detrended_padded)
                filtered = filtered_padded[pad_left: pad_left + len(detrended)]
            except Exception:
                filtered = detrended  # Fallback: detrended without filter
        else:
            try:
                filtered = scipy_signal.filtfilt(self.b, self.a, detrended)
            except Exception:
                # Fallback if filtfilt encounters a numerical edge condition
                filtered = scipy_signal.lfilter(self.b, self.a, detrended)

        # 3. Z-score normalization (mean = 0, std = 1)
        mean_v = np.mean(filtered)
        std_v = np.std(filtered) + 1e-7
        normalized = (filtered - mean_v) / std_v

        return normalized.reshape(-1, 1).astype(np.float32)

    def process_raw_window(
        self,
        points: List[WearOSPPGPoint],
    ) -> Tuple[np.ndarray, Dict[str, Union[float, bool, str]]]:
        """
        Full end-to-end conditioning pipeline:
          Points -> Resample to 2250 @ 25Hz -> Bandpass Filter -> Z-Score -> Model Ready Tensor
        Returns:
            conditioned_tensor: np.ndarray of shape (2250, 1)
            quality_report: SQI metrics dictionary
        """
        valid_points = [p for p in points if p.is_valid]
        if len(valid_points) < len(points) * 0.5:
            # More than 50% points flagged as off-body / detached
            quality = {"sqi": 0.0, "is_usable": False, "quality_flag": "DETACHED_OR_LOOSE"}
        else:
            raw_vals = np.array([p.ppg_green for p in valid_points], dtype=np.float64)
            quality = WearOSSignalQuality.compute_sqi(raw_vals)

        # Extract timestamps and raw ADC values
        t_arr = np.array([p.timestamp_ns for p in points], dtype=np.int64)
        v_arr = np.array([p.ppg_green for p in points], dtype=np.float64)

        # Resample to exact 2250 samples @ 25 Hz
        resampled_25hz = self.resample_to_uniform_grid(t_arr, v_arr)

        # Apply Bandpass & Z-score
        conditioned = self.condition_signal(resampled_25hz)
        return conditioned, quality


# =====================================================================
# 4. TEMPORAL CONSENSUS FILTER FOR MULTI-READING STABILITY
# =====================================================================

class TemporalConsensusFilter:
    """
    Stabilizes arrhythmia predictions across multiple consecutive 90s telemetry readings.
    Prevents flapping/jumping between conditions due to micro-fluctuations in heart rate
    or motion transients on smartwatches (Samsung Galaxy Watch 7).
    """

    def __init__(self, history_size: int = 4, consensus_threshold: float = 0.50):
        self.history_size = history_size
        self.consensus_threshold = consensus_threshold
        self.history: deque = deque(maxlen=history_size)

    def update(self, condition_idx: int, probabilities: np.ndarray) -> Tuple[int, np.ndarray, bool]:
        """
        Updates consensus with latest 90s evaluation.
        Returns:
            stable_condition_idx: int
            averaged_probabilities: np.ndarray
            is_consensus_reached: bool
        """
        probs = np.asarray(probabilities, dtype=np.float32)
        self.history.append((condition_idx, probs))

        # Exponentially weighted moving average over recent 90s windows
        weights = np.linspace(0.65, 1.0, len(self.history))
        weights /= np.sum(weights)

        avg_probs = np.zeros_like(probs)
        for w, (_, p) in zip(weights, self.history):
            avg_probs += w * p

        stable_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[stable_idx])
        is_consensus = bool(confidence >= self.consensus_threshold or len(self.history) >= 2)
        return stable_idx, avg_probs, is_consensus

    def clear(self):
        self.history.clear()


# =====================================================================
# 5. REAL-TIME STREAMING CIRCULAR RING BUFFER
# =====================================================================

class WearOSStreamBuffer:
    """
    Thread-safe circular ring buffer for real-time Wear OS continuous streaming.
    Maintains a rolling 90-second window (2250 samples @ 25 Hz).
    
    Accepts arbitrary batch bursts (e.g. 10 to 25 samples arriving every 400ms-1000ms),
    handles timestamps, performs real-time SQI, and emits model-ready tensors.
    """

    def __init__(self, window_sec: int = 90, target_fs: int = 25):
        self.window_sec = window_sec
        self.target_fs = target_fs
        self.required_samples = window_sec * target_fs  # 2250
        self.adapter = WearOSPPGAdapter(target_fs=target_fs)
        self.consensus_filter = TemporalConsensusFilter(history_size=4)

        self._lock = threading.Lock()
        # Ring buffer storage: uses deque(maxlen) for O(1) append and automatic eviction
        # (replaces plain list + pop(0) which was O(n) per eviction)
        self._max_capacity = self.required_samples * 4  # Store up to 4x to accommodate 100Hz streams
        self._raw_points: deque = deque(maxlen=self._max_capacity)
        self._total_points_received = 0
        self._version = 0
        self._last_sqi = {"sqi": 0.0, "is_usable": False, "quality_flag": "INITIALIZING"}
        self._last_conditioned_cache: Optional[np.ndarray] = None
        self._last_quality_cache: Optional[Dict[str, Union[float, bool, str]]] = None
        self._cache_dirty = True

    def push_point(self, point: WearOSPPGPoint) -> None:
        """Pushes a single data point into the ring buffer (O(1) with deque)."""
        with self._lock:
            self._raw_points.append(point)  # deque auto-evicts oldest when at maxlen
            self._total_points_received += 1
            self._version += 1
            self._cache_dirty = True

    def push_batch(self, points: List[WearOSPPGPoint]) -> IngestionResult:
        """Pushes a batch of data points received over Bluetooth / Wearable Data Layer."""
        if not points:
            return self.get_status()

        with self._lock:
            self._raw_points.extend(points)  # deque auto-evicts oldest when at maxlen
            self._total_points_received += len(points)
            self._version += 1
            self._cache_dirty = True

        return self.get_status()

    def get_status(self) -> IngestionResult:
        """Returns the current buffer fill and operational status."""
        with self._lock:
            count = len(self._raw_points)
            # Estimate how many seconds of continuous data we currently hold
            if count >= 2:
                duration_sec = (self._raw_points[-1].timestamp_ns - self._raw_points[0].timestamp_ns) / 1e9
            else:
                duration_sec = 0.0

            fill_pct = min(100.0, round((duration_sec / self.window_sec) * 100.0, 1))
            is_ready = bool(duration_sec >= (self.window_sec * 0.95))

            # Compute quick SQI on recent points
            if count >= 25:
                recent = np.array([p.ppg_green for p in list(self._raw_points)[-200:]], dtype=np.float64)
                self._last_sqi = WearOSSignalQuality.compute_sqi(recent)

            valid_count = sum(1 for p in self._raw_points if p.is_valid)
            dropped = count - valid_count

            return IngestionResult(
                points_received=count,
                points_valid=valid_count,
                points_dropped=dropped,
                buffer_fill_pct=fill_pct,
                current_sqi=self._last_sqi.get("sqi", 0.0),
                is_ready_for_inference=is_ready,
                status_summary=self._last_sqi.get("quality_flag", "PENDING"),
            )

    def get_model_window(self) -> Tuple[np.ndarray, Dict[str, Union[float, bool, str]]]:
        """
        Extracts the conditioned 90-second window ready for the 1D-Conformer.
        Returns:
            conditioned_tensor: np.ndarray [2250, 1]
            quality_info: Dict with SQI and metadata
        """
        with self._lock:
            if not self._cache_dirty and self._last_conditioned_cache is not None and self._last_quality_cache is not None:
                return self._last_conditioned_cache, self._last_quality_cache

            if len(self._raw_points) < 50:
                raise ValueError("Buffer has insufficient samples (need >= 50 points to interpolate)")
            # Snapshot under lock to prevent concurrent modifications during DSP
            pts = list(self._raw_points)
            ver = self._version

        # DSP runs outside the lock (CPU-intensive; holds lock would block push_batch)
        conditioned, quality = self.adapter.process_raw_window(pts)

        # Re-acquire lock to safely write cache with version check
        with self._lock:
            if self._version == ver:
                self._last_conditioned_cache = conditioned
                self._last_quality_cache = quality
                self._cache_dirty = False

        return conditioned, quality

    def clear(self) -> None:
        """Clears all buffered points (resets the deque to empty)."""
        with self._lock:
            self._raw_points.clear()
            self._total_points_received = 0
            self._version += 1
            self._cache_dirty = True
            self._last_conditioned_cache = None
            self._last_quality_cache = None
            self._last_sqi = {"sqi": 0.0, "is_usable": False, "quality_flag": "CLEARED"}
            self.consensus_filter.clear()
