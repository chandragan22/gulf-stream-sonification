"""
Gulf Stream Surface Currents — Shimmering-Cluster Sonification for Alchemy
============================================================================
Six held notes forming a D-centered cluster, played on a single Alchemy
instrument in Logic.  Lower three notes form a stable bed; upper three
shimmer in and out based on the smaller EOF modes.

Cluster
-------
  Bed      D4 (62)  E4 (64)  A4 (69)     — always present
  Shimmer  B4 (71)  D5 (74)  F#5 (78)    — modulated by PC2, PC3, PC4

All six notes share the same global modulation (filter, reverb, chorus,
pitch drift, volume), so the cluster moves as one sonic body.
The shimmer notes additionally have their own per-voice expression
(CC11) tracking their assigned PC — when that mode is weak, the shimmer
recedes; when strong, it intensifies.

Modulation map
--------------
Global (all channels):
  PC1 (slow, dominant)          → CC74  filter cutoff   — breathing
  Basin-mean speed              → CC7   main volume     — energy
  Eddy fraction (Okubo-Weiss)   → CC91  reverb send     — spatial swirl
  Slow-filtered PC1             → pitch bend (±50 cents)

Per-channel shimmer modulation (shimmer notes only):
  PC2  → CC11 on B4 channel
  PC3  → CC11 on D5 channel
  PC4  → CC11 on F#5 channel

Bed notes get a gentle, stable CC11 curve so they remain as the
harmonic foundation.

Output: currents_alchemy.mid — 6-track MIDI for one Alchemy instrument.
"""

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

# ── Configuration ─────────────────────────────────────────────────────────────
FILE = "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i_1775676050764.nc"
OUTFILE = "currents_alchemy.mid"

BPM = 120
TICKS_PER_BEAT = 480
TOTAL_FRAMES = 1440           # 12 min at 120 BPM (1 frame = 1 beat = 0.5 sec)
TICKS_PER_FRAME = TICKS_PER_BEAT
PITCH_BEND_RANGE = 2048       # ±50 cents (default MIDI bend range is ±200 cents)

# Cluster voicing.  Each dict: MIDI note, role, and (for shimmer) which PC
# drives the per-channel CC11 expression.
# All six voices go to the same Alchemy instrument in Logic — assign all
# six MIDI tracks to the same software instrument on import.
VOICES = [
    {"note": 62, "role": "bed",     "pc_idx": None, "name": "D4"},
    {"note": 64, "role": "bed",     "pc_idx": None, "name": "E4"},
    {"note": 69, "role": "bed",     "pc_idx": None, "name": "A4"},
    {"note": 71, "role": "shimmer", "pc_idx": 1,    "name": "B4  (PC2)"},
    {"note": 74, "role": "shimmer", "pc_idx": 2,    "name": "D5  (PC3)"},
    {"note": 78, "role": "shimmer", "pc_idx": 3,    "name": "F#5 (PC4)"},
]

# ── 1. Load and preprocess ────────────────────────────────────────────────────
print("Loading CMEMS data...")
ds = xr.open_dataset(FILE)
U = ds["utotal"].squeeze("depth").values
V = ds["vtotal"].squeeze("depth").values
lat = ds["latitude"].values
lon = ds["longitude"].values

speed = np.sqrt(U**2 + V**2)
N_t, N_lat, N_lon = speed.shape
print(f"  shape: {N_t} hrs × {N_lat} lat × {N_lon} lon")

# ── 2. Basin-mean speed ───────────────────────────────────────────────────────
basin_speed = np.nanmean(speed, axis=(1, 2))

# ── 3. Okubo-Weiss eddy fraction ──────────────────────────────────────────────
R_EARTH = 6.371e6
dy_m = R_EARTH * np.deg2rad(float(lat[1] - lat[0]))
dx_m = R_EARTH * np.cos(np.deg2rad(lat)) * np.deg2rad(float(lon[1] - lon[0]))

print("Computing Okubo-Weiss...")
dudy = np.gradient(U, dy_m, axis=1)
dvdy = np.gradient(V, dy_m, axis=1)
dudx = np.gradient(U, axis=2) / dx_m[np.newaxis, :, np.newaxis]
dvdx = np.gradient(V, axis=2) / dx_m[np.newaxis, :, np.newaxis]
Sn, Ss = dudx - dvdy, dvdx + dudy
zeta = dvdx - dudy
W = Sn**2 + Ss**2 - zeta**2
sigma_W_t = np.nanstd(W, axis=(1, 2))
eddy_fraction = np.nanmean(W < -0.2 * sigma_W_t[:, None, None], axis=(1, 2))

# ── 4. EOFs ───────────────────────────────────────────────────────────────────
print("Computing EOFs...")
speed_2d = speed.reshape(N_t, N_lat * N_lon)
speed_2d = np.where(np.isnan(speed_2d), 0.0, speed_2d)
speed_anom = speed_2d - speed_2d.mean(axis=0)

pca = PCA(n_components=4)
PCs = pca.fit_transform(speed_anom)
var = pca.explained_variance_ratio_
for k in range(4):
    print(f"  PC{k+1}: {var[k]*100:5.2f}% of variance")

# ── 5. Resample everything to TOTAL_FRAMES ────────────────────────────────────
def resample(x, n_out):
    return np.interp(np.linspace(0, 1, n_out),
                     np.linspace(0, 1, len(x)), x)

print(f"\nResampling {N_t} hours → {TOTAL_FRAMES} frames "
      f"({TOTAL_FRAMES*60/BPM/60:.1f} min)")
PC = np.column_stack([resample(PCs[:, k], TOTAL_FRAMES) for k in range(4)])
basin = resample(basin_speed, TOTAL_FRAMES)
eddy = resample(eddy_fraction, TOTAL_FRAMES)

# ── 6. Mapping helpers ────────────────────────────────────────────────────────
def normalize(x, lo_pct=2, hi_pct=98):
    lo = np.nanpercentile(x, lo_pct)
    hi = np.nanpercentile(x, hi_pct)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

def to_cc(x, cc_min=0, cc_max=127):
    return (cc_min + normalize(x) * (cc_max - cc_min)).astype(int)

# Global CCs (applied to every channel so Alchemy moves as one)
#   CC74 (filter cutoff): avoid fully closed (silent) and fully open (harsh)
cc74_global = to_cc(PC[:, 0], cc_min=30, cc_max=110)
#   CC7  (main volume): keep bed always audible
cc7_global = to_cc(basin, cc_min=70, cc_max=120)
#   CC91 (reverb send): significant wet-signal range
cc91_global = to_cc(eddy, cc_min=25, cc_max=105)

# Per-channel CC11 (expression) for each voice:
#   Bed voices get a flat-ish curve — gentle drift, always present.
#   Shimmer voices track their assigned PC — can fade nearly to silence
#     when their mode is weak, and bloom up when it is strong.
voice_cc11 = []
for v in VOICES:
    if v["role"] == "bed":
        # Very gentle expression drift driven by basin_speed — 90 to 115.
        voice_cc11.append(to_cc(basin, cc_min=90, cc_max=115))
    else:
        # Shimmer: 30 (barely audible) to 125 (full bloom) based on |PC|.
        # We use |PC - median| so both positive and negative excursions bloom.
        pc = PC[:, v["pc_idx"]]
        magnitude = np.abs(pc - np.median(pc))
        voice_cc11.append(to_cc(magnitude, cc_min=30, cc_max=125))

# ── 7. Pitch bend: slow-filtered PC1 → ±50 cents drift ───────────────────────
b, a = butter(N=3, Wn=0.02, btype="low")
pc1_smooth = filtfilt(b, a, PC[:, 0])
pc1_norm = normalize(pc1_smooth) * 2 - 1     # -1 to +1
pitch_bend_values = (pc1_norm * PITCH_BEND_RANGE).astype(int)

# ── 8. Build MIDI ─────────────────────────────────────────────────────────────
print("\nBuilding MIDI...")
mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)

# Track 0: conductor (tempo marker only — no notes)
conductor = MidiTrack()
conductor.append(MetaMessage("set_tempo", tempo=bpm2tempo(BPM), time=0))
conductor.append(MetaMessage("track_name", name="Conductor", time=0))
mid.tracks.append(conductor)

# One MIDI track per voice.  Each track uses its own channel (0-5) so that
# per-channel CC11 doesn't cross-contaminate.  All tracks will be routed to
# the SAME Alchemy instrument in Logic.
for v_idx, voice in enumerate(VOICES):
    track = MidiTrack()
    channel = v_idx
    track.append(MetaMessage("track_name", name=voice["name"], time=0))
    mid.tracks.append(track)

    # Start note with modest velocity — CC11/CC7 drive the dynamics
    initial_vel = 75 if voice["role"] == "bed" else 55
    track.append(Message("note_on", channel=channel, note=voice["note"],
                         velocity=initial_vel, time=0))

    # Emit CC + pitch bend updates each frame, with de-duplication
    prev = {"74": -1, "7": -1, "91": -1, "11": -1, "bend": -1}
    last_tick = 0
    cur_tick = 0
    v_cc11 = voice_cc11[v_idx]

    for f in range(TOTAL_FRAMES):
        updates = [
            ("74", 74, int(cc74_global[f])),
            ("7",   7, int(cc7_global[f])),
            ("91", 91, int(cc91_global[f])),
            ("11", 11, int(v_cc11[f])),
        ]
        for key, cc_num, val in updates:
            if val != prev[key]:
                delta = cur_tick - last_tick
                track.append(Message("control_change", channel=channel,
                                     control=cc_num, value=val, time=delta))
                last_tick = cur_tick
                prev[key] = val

        pb = int(pitch_bend_values[f])
        if pb != prev["bend"]:
            delta = cur_tick - last_tick
            track.append(Message("pitchwheel", channel=channel,
                                 pitch=pb, time=delta))
            last_tick = cur_tick
            prev["bend"] = pb

        cur_tick += TICKS_PER_FRAME

    # Close the held note
    delta = cur_tick - last_tick
    track.append(Message("note_off", channel=channel, note=voice["note"],
                         velocity=0, time=delta))

    print(f"  {voice['name']:<14} ({voice['role']:<7}) channel {channel}: "
          f"{len(track)} events")

# ── 9. Save ───────────────────────────────────────────────────────────────────
mid.save(OUTFILE)
duration_min = TOTAL_FRAMES * 60 / BPM / 60
print(f"\nSaved {OUTFILE}")
print(f"Duration: {duration_min:.1f} min")

# ── Logic setup notes ─────────────────────────────────────────────────────────
print("""
Logic Pro setup:

1. Import the MIDI:  File → Import → MIDI → currents_alchemy.mid
   You'll get 6 tracks (plus the conductor which has no audio).

2. IMPORTANT — route all 6 tracks to ONE Alchemy instrument:
   - Create one Software Instrument track with Alchemy
   - For each imported MIDI track, set its output to that same Alchemy
     instrument.  In the inspector, change each track's instrument to
     "No Output" on its own channel strip, then drag its MIDI region
     onto the single Alchemy track, OR use Logic's "MIDI Thru" routing.

   Simpler alternative: after import, select all 6 regions and drag them
   onto a single Alchemy track.  Logic will play all 6 through that
   single instance.

3. Pick an Alchemy preset to start.  Recommended categories/names:
     - Pad → Ocean Current, Deep Waters, Submerged
     - Pad → Morphing, Evolving, Glacial
     - Texture → Aqua, Liquid, Atmospheric
   The MIDI will animate whichever preset you choose.

4. The key CCs are:
     CC74  filter cutoff  (global, PC1 — the main breathing)
     CC7   volume         (global, basin speed — energy level)
     CC91  reverb send    (global, eddy fraction — spatial bloom)
     CC11  expression     (per voice — the shimmer layer)
     Pitch bend           (global, slow-drifted PC1 — alive but not melodic)

5. If any CC isn't audibly doing anything, check Alchemy's Perform
   controls — some presets don't expose filter cutoff to CC74 out of
   the box.  Fix: open Alchemy, click the small controller icon next
   to any knob, and MIDI-learn it to the incoming CC.

Expectation:
  - Shimmer notes (B4, D5, F#5) should be barely present during calm
    periods and bloom during active eddy/meander periods.
  - Bed notes (D4, E4, A4) should be continuously present — they're
    the anchor.
  - The whole cluster should feel like it's breathing together,
    not six independent voices.
""")
