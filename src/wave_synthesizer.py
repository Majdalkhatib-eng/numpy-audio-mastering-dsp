import numpy as np

class SubHarmonicSynthesizer:
    """
    Sub-Bass Waveform Generator using Continuous Phase Numerical Integration
    and Exponential Frequency Decays.
    """
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

    def synthesize_sub_drop(self, duration_sec=1.6, f_start=75.0, f_end=28.0, decay_rate=1.4):
        """
        Synthesizes a clean transient sub-bass sweep without phase discontinuities.
        Equation: f(t) = (f_start - f_end) * exp(-decay * t) + f_end
        """
        n_samp = int(duration_sec * self.sr)
        t = np.linspace(0.0, duration_sec, n_samp, dtype=np.float32)
        
        # التردد التنازلي الأسي
        freq_curve = (f_start - f_end) * np.exp(-t * decay_rate) + f_end
        
        # التكامل العددي لحساب الطور التراكمي phi(t) لمنع تشوه الطور
        phase = 2.0 * np.pi * np.cumsum(freq_curve) / float(self.sr)
        
        # الموجة الجيبية مع التخميد الأسي
        wave = np.sin(phase) * np.exp(-t * 2.2)
        
        # تلاشي تدريجي لمنع الـ Clicks
        fade_in = min(n_samp, int(0.04 * self.sr))
        wave[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)

        stereo_wave = np.column_stack([wave, wave]) * 0.5
        return stereo_wave.astype(np.float32)