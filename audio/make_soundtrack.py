"""Synthesize the soundtrack for the Seismic "Encrypted" trailer.

Everything is generated from code (no samples), following the same timeline
as seismic-trailer.html and blender/seismic_trailer.py.

    python3 audio/make_soundtrack.py            # writes audio/seismic-trailer.wav
    ffmpeg -i audio/seismic-trailer.wav -b:a 160k audio/seismic-trailer.mp3

Requires only numpy.
"""
import os
import wave

import numpy as np

SR = 44100
DUR = 48.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(7)

L = np.zeros(N)
R = np.zeros(N)
send = np.zeros(N)  # reverb bus (mono)


def idx(sec):
    return max(0, min(N, int(sec * SR)))


def place(sig, at, gain=1.0, pan=0.0, verb=0.0):
    """Mix a mono signal into the stereo bus starting at `at` seconds."""
    a = idx(at)
    b = min(N, a + len(sig))
    s = sig[: b - a] * gain
    L[a:b] += s * np.sqrt(0.5 * (1 - pan))
    R[a:b] += s * np.sqrt(0.5 * (1 + pan))
    send[a:b] += s * verb


def band(x, lo, hi):
    """Brick-wall band-pass via FFT (fine for noise beds)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def env_ad(n, attack, decay_tau):
    tt = np.arange(n) / SR
    e = np.exp(-np.maximum(0, tt - attack) / decay_tau)
    if attack > 0:
        e *= np.clip(tt / attack, 0, 1)
    return e


def win(a, b, fi=0.3, fo=0.3):
    """Smooth gate over the global timeline."""
    up = np.clip((t - a) / fi, 0, 1)
    dn = np.clip((b - t) / fo, 0, 1)
    return np.minimum(up, dn) * ((t >= a) & (t <= b))


def sweep(f0, f1, dur, shape="exp"):
    n = int(dur * SR)
    k = np.linspace(0, 1, n)
    f = f0 * (f1 / f0) ** k if shape == "exp" else f0 + (f1 - f0) * k
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def saw(freq, n, harmonics=8):
    tt = np.arange(n) / SR
    return sum(np.sin(2 * np.pi * freq * h * tt) / h for h in range(1, harmonics + 1))


def note(name):
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    pitch, octave = name[:-1], int(name[-1])
    return 440.0 * 2 ** ((names[pitch] + 12 * (octave - 4)) / 12)


# ---------------- building blocks ----------------
def kick(gain=1.0):
    n = int(0.45 * SR)
    return sweep(150, 42, 0.45) * env_ad(n, 0.002, 0.11) * gain


def snare():
    n = int(0.22 * SR)
    return (band(rng.standard_normal(n), 900, 9000) * 0.6 + np.sin(2 * np.pi * 190 * np.arange(n) / SR) * 0.5) * env_ad(n, 0.001, 0.06)


def hat():
    n = int(0.05 * SR)
    return band(rng.standard_normal(n), 6000, 16000) * env_ad(n, 0.0005, 0.012)


def bell(f, dur=2.4):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    parts = [(1, 1.0), (2.0, 0.5), (3.01, 0.3), (4.2, 0.18), (5.4, 0.1)]
    return sum(a * np.sin(2 * np.pi * f * m * tt) * np.exp(-tt * (1.5 + m)) for m, a in parts)


def clank():
    n = int(0.5 * SR)
    tt = np.arange(n) / SR
    fs = rng.uniform(400, 2600, 4)
    return sum(np.sin(2 * np.pi * f * tt) for f in fs) * np.exp(-tt * 9) * 0.25


def boom(f0=80, f1=26, dur=2.6):
    n = int(dur * SR)
    body = sweep(f0, f1, dur) * env_ad(n, 0.004, 0.75)
    crack = band(rng.standard_normal(n), 60, 3500) * env_ad(n, 0.001, 0.18)
    return np.tanh((body * 1.4 + crack * 0.8) * 1.6)


def rumble(dur, lo=20, hi=140):
    n = int(dur * SR)
    return band(rng.standard_normal(n), lo, hi)


def pad(freqs, dur, attack=0.8, release=1.2, bright=6):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    x = sum(saw(f, n, bright) + saw(f * 1.004, n, bright) for f in freqs) / (2 * len(freqs))
    e = np.clip(tt / attack, 0, 1) * np.clip((dur - tt) / release, 0, 1)
    return x * e


# ---------------- 0–48s timeline ----------------
# rain bed (decorrelated L/R); muffled during the blackout
rain_amt = 1 - 0.75 * win(31.0, 32.7, 0.15, 0.6)
rain_hi_l = band(rng.standard_normal(N), 900, 9000)
rain_hi_r = band(rng.standard_normal(N), 900, 9000)
rain_lo = band(rng.standard_normal(N), 120, 700)
fade_in = np.clip(t / 2.0, 0, 1)
L += (rain_hi_l * 0.035 * rain_amt + rain_lo * 0.03) * fade_in
R += (rain_hi_r * 0.035 * rain_amt + rain_lo * 0.03) * fade_in

# tense D-minor drone under the chase and siege
drone = pad([note("D2"), note("A2"), note("D3"), note("F3")], 30.6, attack=3.0, release=0.4, bright=5)
lfo = 0.75 + 0.25 * np.sin(2 * np.pi * 0.25 * np.arange(len(drone)) / SR)
place(drone * lfo, 0.0, gain=0.22, verb=0.25)
place(pad([note("A#1"), note("E3"), note("A#2")], 8.0, attack=2.0, release=0.5), 16.2, gain=0.12, verb=0.3)

# chase drums, 128 bpm
beat = 60 / 128
b = 4.6
while b < 15.9:
    i = round((b - 4.6) / beat)
    place(kick(), b, gain=0.55)
    if i % 2 == 1:
        place(snare(), b, gain=0.32, verb=0.15)
    place(hat(), b, gain=0.18, pan=0.3)
    place(hat(), b + beat / 2, gain=0.12, pan=-0.3)
    b += beat
place(sweep(400, 2400, 0.9) * env_ad(int(0.9 * SR), 0.6, 0.15), 15.0, gain=0.06)

# heartbeat during the siege, speeding up
hb = 16.4
while hb < 24.2:
    rate = 72 + (hb - 16.4) * 5
    place(kick(0.8), hb, gain=0.5)
    place(kick(0.6), hb + 0.18, gain=0.32)
    hb += 60 / rate

# MEV drones: detuned buzz while they are in the air
dn = idx(30.4) - idx(4.0)
buzz = band(saw(176, dn, 10) + saw(181.5, dn, 10), 300, 3000)
buzz *= 0.6 + 0.4 * np.sin(2 * np.pi * 7 * np.arange(dn) / SR)
buzz *= np.clip(np.arange(dn) / SR / 1.5, 0, 1)
place(buzz, 4.0, gain=0.035, pan=0.2)

# scanner beeps while they read the transaction
for k in range(10):
    bt = 8.45 + k * 0.24
    n = int(0.06 * SR)
    place(np.sign(np.sin(2 * np.pi * 1760 * np.arange(n) / SR)) * env_ad(n, 0.001, 0.03), bt, gain=0.05, pan=0.4)

# White: glasses glint, the throw, the catch
place(bell(3520, 0.8), 11.3, gain=0.05, pan=-0.5, verb=0.6)
place(bell(3520, 0.8), 43.3, gain=0.05, pan=-0.3, verb=0.6)
wn = int(2.45 * SR)
whoosh = band(rng.standard_normal(wn), 500, 5000) * np.sin(np.linspace(0, np.pi, wn)) ** 2
place(whoosh, 13.0, gain=0.12, pan=0.2, verb=0.2)
for f, g in [(note("D6"), 0.12), (note("A6"), 0.08), (note("F6"), 0.06)]:
    place(bell(f, 2.6), 15.45, gain=g, verb=0.7)

# planting the crystal: impact, shimmer, spreading cracks
place(boom(90, 40, 1.6), 21.0, gain=0.45, verb=0.35)
for k, f in enumerate([note("D5"), note("F5"), note("A5"), note("D6"), note("E6")]):
    place(bell(f, 2.0), 21.05 + k * 0.08, gain=0.05, verb=0.8)
cr = rumble(3.4, 25, 160) * np.linspace(0.2, 1, int(3.4 * SR))
place(cr, 21.0, gain=0.35)
for k in range(40):
    ct = 21.2 + rng.uniform(0, 3.1)
    n = int(0.03 * SR)
    place(band(rng.standard_normal(n), 1500, 7000) * env_ad(n, 0.0005, 0.008), ct, gain=0.12, pan=rng.uniform(-0.8, 0.8))

# the golem rises: earthquake rumble, rock crunches, eyes ignite
place(rumble(3.6, 18, 120) * np.sin(np.linspace(0, np.pi, int(3.6 * SR))) ** 0.5, 24.3, gain=0.9)
for k in range(26):
    ct = 24.4 + rng.uniform(0, 3.0)
    n = int(0.09 * SR)
    place(np.tanh(band(rng.standard_normal(n), 80, 2500) * 3) * env_ad(n, 0.001, 0.03), ct, gain=0.22, pan=rng.uniform(-0.7, 0.7), verb=0.2)
place(sweep(180, 1400, 0.5) * env_ad(int(0.5 * SR), 0.35, 0.1), 26.45, gain=0.09, verb=0.5)
place(boom(60, 30, 1.8), 26.5, gain=0.35)
place(pad([note("D2"), note("A2"), note("D3")], 3.6, attack=0.4, release=1.0, bright=9), 26.6, gain=0.18, verb=0.4)

# wind-up riser into the punch
rn = int(1.95 * SR)
riser = band(rng.standard_normal(rn), 1500, 12000) * np.linspace(0, 1, rn) ** 2
riser += sweep(90, 420, 1.95) * np.linspace(0, 1, rn) ** 2 * 0.6
place(riser, 28.25, gain=0.2, verb=0.3)

# IMPACT
place(boom(85, 24, 3.2), 30.2, gain=1.0, verb=0.5)
place(boom(140, 50, 1.0), 30.2, gain=0.5, pan=0.3)
for k in range(12):
    place(clank(), 30.45 + rng.uniform(0, 1.4), gain=0.5, pan=rng.uniform(-0.9, 0.9), verb=0.3)

# blackout: ringing ears
tn = int(1.8 * SR)
place(np.sin(2 * np.pi * 6200 * np.arange(tn) / SR) * env_ad(tn, 0.05, 0.6), 30.9, gain=0.02)

# encrypted: the city powers back on in digital blips
place(sweep(60, 900, 0.6) * env_ad(int(0.6 * SR), 0.5, 0.1), 32.55, gain=0.15, verb=0.4)
scale = [note(n) for n in ["D5", "F5", "G5", "A5", "C6", "D6", "F6", "A6"]]
for k in range(40):
    bt = 32.7 + k * 0.085
    f = scale[(k * 5 + (k // 4)) % len(scale)]
    n = int(0.07 * SR)
    blip = np.sign(np.sin(2 * np.pi * f * np.arange(n) / SR)) * env_ad(n, 0.001, 0.025)
    place(blip, bt, gain=0.035 * (1 - k / 48), pan=0.6 * np.sin(k), verb=0.4)
place(pad([note("D3"), note("A3"), note("F4")], 8.5, attack=1.5, release=1.5, bright=4), 32.8, gain=0.12, verb=0.5)

# White jumps down and lands
jn = int(1.6 * SR)
place(band(rng.standard_normal(jn), 300, 4000) * np.sin(np.linspace(0, np.pi, jn)) ** 2, 36.0, gain=0.12, pan=-0.3)
place(boom(110, 40, 1.2), 37.6, gain=0.5, pan=-0.2, verb=0.3)

# hero theme: Dm – Bb – F – C, then the logo hit
prog = [(["D3", "F3", "A3", "D2"], 41.0), (["A#2", "D3", "F3", "A#1"], 42.75), (["F3", "A3", "C4", "F2"], 44.5)]
for chord, at in prog:
    place(pad([note(n) for n in chord], 1.95, attack=0.3, release=0.6, bright=7), at, gain=0.2, verb=0.45)
hb = 42.6
while hb < 45.5:
    place(kick(0.9), hb, gain=0.5)
    hb += 60 / 90
place(boom(70, 30, 3.0), 45.6, gain=0.75, verb=0.6)
place(pad([note(n) for n in ["F2", "C3", "F3", "A3", "C4", "E4"]], 2.4, attack=0.05, release=2.0, bright=8), 45.6, gain=0.3, verb=0.7)
for k, f in enumerate([note("C6"), note("E6"), note("A6"), note("C7")]):
    place(bell(f, 2.4), 45.65 + k * 0.12, gain=0.06, verb=0.8)

# ---------------- reverb + master ----------------
ir_n = int(2.2 * SR)
ir = rng.standard_normal(ir_n) * np.exp(-np.arange(ir_n) / SR / 0.55)
ir = band(ir, 200, 9000)
ir /= np.sqrt(np.sum(ir ** 2))
size = 1 << int(np.ceil(np.log2(N + ir_n)))
wet = np.fft.irfft(np.fft.rfft(send, size) * np.fft.rfft(ir, size), size)[:N]
L += wet * 0.5
R += wet * 0.5

mix = np.stack([L, R], axis=1)
mix *= np.clip((DUR - t) / 1.5, 0, 1)[:, None]  # tail fade
mix = np.tanh(mix * 1.2) / np.tanh(1.2)
mix /= np.max(np.abs(mix)) / 0.89

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seismic-trailer.wav")
with wave.open(out, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype("<i2").tobytes())
print("wrote", out)
