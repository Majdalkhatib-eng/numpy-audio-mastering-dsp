import numpy as np

class LookaheadAudioDucker:
    """
    Acoustic-Aware Audio Ducking Engine featuring RMS Energy Detection,
    Asymmetric Attack/Decay Lookahead Tracking, and Convolution Smoothing.
    """
    def __init__(self, sample_rate=44100, window_ms=25):
        self.sr = sample_rate
        self.win_len = int((window_ms / 1000.0) * sample_rate)

    def generate_ducking_envelope(self, voice_signal, lookahead_steps=3):
        total_samples = len(voice_signal)
        num_wins = max(1, total_samples // self.win_len)

        # 1. حساب الـ RMS لكل نافذة زمنية
        v_mono = np.mean(voice_signal, axis=1) if voice_signal.ndim > 1 else voice_signal
        reshaped = v_mono[:num_wins * self.win_len].reshape(num_wins, self.win_len)
        rms_per_win = np.sqrt(np.mean(reshaped ** 2, axis=1))

        # عتبة الكلام النشط
        voice_gate = (rms_per_win > 0.018).astype(np.float32)

        # 2. متتبع الغلاف غير المتناظر مع الاستشراف (Lookahead)
        env_smooth = np.zeros_like(voice_gate)
        cur_env = 0.0
        for i in range(num_wins):
            la_val = np.max(voice_gate[i:min(num_wins, i + lookahead_steps)])
            if la_val > cur_env:
                cur_env = 0.65 * la_val + 0.35 * cur_env  # Attack سريع
            else:
                cur_env = 0.065 * la_val + 0.935 * cur_env # Decay ناعم
            env_smooth[i] = cur_env

        # 3. تنعيم رياضي عبر الالتفاف (Convolution Smoothing)
        kernel = np.ones(9, dtype=np.float32) / 9.0
        env_smooth = np.convolve(env_smooth, kernel, mode="same")

        # إعادة مطابقة الطول للأصلي بالاستيفاء الخطي
        return np.interp(
            np.linspace(0.0, 1.0, total_samples),
            np.linspace(0.0, 1.0, num_wins),
            env_smooth
        ).astype(np.float32)

    def duck_music(self, bgm_signal, ducking_env, ducked_gain=0.15, normal_gain=0.85):
        """
        يطبق الغلاف على موسيقى الخلفية
        """
        gain_curve = (1.0 - ducking_env) * normal_gain + ducking_env * ducked_gain
        if bgm_signal.ndim > 1:
            gain_curve = gain_curve[:, np.newaxis]
        return bgm_signal * gain_curve