import numpy as np
import soundfile as sf
from src.cathedral_reverb import CathedralReverbDSP
from src.wave_synthesizer import SubHarmonicSynthesizer
from src.dynamic_ducker import LookaheadAudioDucker

def main():
    sr = 44100
    duration = 6.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    print("[*] Generating synthetic test signals...")
    # محاكاة صوت متحدث (نغمات متقطعة)
    voice = np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0.3).astype(np.float32) * 0.4
    voice_stereo = np.column_stack([voice, voice])

    # محاكاة موسيقى مستمرة
    bgm = np.sin(2 * np.pi * 440 * t) * 0.25
    bgm_stereo = np.column_stack([bgm, bgm])

    # 1. اختبار محاكي صدى الكاتدرائيات
    print("[*] Applying 12-Tap Physical Cathedral Reverb...")
    reverb_engine = CathedralReverbDSP(sample_rate=sr)
    wet_voice = reverb_engine.apply_reverb(voice_stereo)

    # 2. اختبار الـ Sub-Drop
    print("[*] Synthesizing Sub-Bass sweep pulse...")
    sub_synth = SubHarmonicSynthesizer(sample_rate=sr)
    sub_pulse = sub_synth.synthesize_sub_drop(duration_sec=2.0)

    # 3. اختبار الـ Dynamic Ducking
    print("[*] Executing Lookahead Ducking & FIR Smoothing...")
    ducker = LookaheadAudioDucker(sample_rate=sr)
    env = ducker.generate_ducking_envelope(voice_stereo)
    ducked_bgm = ducker.duck_music(bgm_stereo, env)

    # الدمج والتطبيق النهائي للـ Soft Clipping عبر tanh
    mixed = wet_voice + ducked_bgm
    mixed[:len(sub_pulse)] += sub_pulse
    mastered = np.tanh(mixed * 1.1) / 1.1

    out_file = "mastered_dsp_output.wav"
    sf.write(out_file, mastered, sr)
    print(f"[OK] DSP Pipeline executed successfully! Output: {out_file}")

if __name__ == "__main__":
    main()