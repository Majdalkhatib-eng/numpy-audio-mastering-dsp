# NumPy Audio Mastering & Acoustic DSP Engine

A high-fidelity **Digital Signal Processing (DSP)** library written in NumPy for acoustic spatial emulation, procedural sub-bass wave synthesis, and intelligent dynamic audio ducking.

---

## Technical Highlights

* **12-Tap Physical Cathedral Reverb:** Models large stone-hall acoustics using 12 discrete fractional delay lines with stereo cross-channel matrix feeds and Hanning FIR low-pass damping.
* **Continuous Phase Sub-Bass Synthesis:** Generates transient bass sweeps without phase clicks by computing continuous phase trajectories via numerical cumulative summation (`np.cumsum`).
* **Lookahead Energy Ducking:** Employs windowed RMS energy detection combined with an asymmetric lookahead envelope follower and moving-average convolution smoothing.
* **Analog Soft Saturation:** Features hyperbolic tangent limiting (`np.tanh`) to eliminate digital inter-sample clipping while imparting warm acoustic saturation.

---

## Mathematical Formulation

### 1. Cumulative Phase Trajectory
To prevent phase discontinuities during exponential frequency decay sweeps:
$$\phi(t) = 2\pi \sum_{n=0}^{N} \frac{f(n)}{f_s}$$
$$x(t) = \sin(\phi(t)) \cdot e^{-\alpha t}$$

### 2. Lookahead Envelope Generation
$$\text{RMS}_w = \sqrt{\frac{1}{W} \sum_{k=0}^{W-1} x^2[k]}$$
$$\text{Env}[i] = \max_{0 \le j \le L} (\text{Gate}[i + j]) * h_{\text{FIR}}$$

---

## Installation

```bash
git clone [https://github.com/YOUR_USERNAME/numpy-audio-mastering-dsp.git](https://github.com/YOUR_USERNAME/numpy-audio-mastering-dsp.git)
cd numpy-audio-mastering-dsp
pip install -r requirements.txt
```

---

## Quickstart

Run the complete pipeline demonstration:

```bash
python demo.py
```

---

## License

This project is licensed under the MIT License.