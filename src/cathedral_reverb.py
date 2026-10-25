import numpy as np

class CathedralReverbDSP:
    """
    Physical Acoustic Emulation using a 12-Tap Multi-Delay Line with Stereo Cross-Feed
    and Hanning FIR Low-Pass Absorption Damping.
    """
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        # خطوط التأخير الفيزيائية مع معاملات التوزيع الستيريو
        self.taps = [
            (0.019, 0.22, 0.10), (0.033, 0.11, 0.21), (0.051, 0.18, 0.14),
            (0.073, 0.13, 0.16), (0.097, 0.14, 0.11), (0.125, 0.10, 0.13),
            (0.159, 0.11, 0.08), (0.197, 0.08, 0.10), (0.241, 0.07, 0.06),
            (0.293, 0.05, 0.06), (0.353, 0.05, 0.04), (0.421, 0.03, 0.04)
        ]

    def apply_reverb(self, audio_stereo):
        if audio_stereo.ndim == 1:
            audio_stereo = np.column_stack([audio_stereo, audio_stereo])

        seg_len = len(audio_stereo)
        # 1. مرشح هانينغ لمحاكاة امتصاص الجدران الحجرية للترددات العالية (FIR Damping)
        lp_kernel = np.hanning(19).astype(np.float32)
        lp_kernel /= np.sum(lp_kernel)
        damped_seg = np.column_stack([
            np.convolve(audio_stereo[:, 0], lp_kernel, mode="same"),
            np.convolve(audio_stereo[:, 1], lp_kernel, mode="same")
        ])

        # 2. مصفوفة الصدى متعدد التفرعات مع التغذية المتقاطعة (Cross-feed)
        wet = np.zeros_like(audio_stereo, dtype=np.float32)
        for delay_s, g_l, g_r in self.taps:
            d_samp = int(delay_s * self.sr)
            if d_samp < seg_len:
                wet[d_samp:, 0] += damped_seg[:-d_samp, 1] * g_l
                wet[d_samp:, 1] += damped_seg[:-d_samp, 0] * g_r

        # خلط الإشارة الأصلية الجافة (Dry) مع الإشارة الرطبة (Wet)
        return audio_stereo * 1.0 + wet * 0.85