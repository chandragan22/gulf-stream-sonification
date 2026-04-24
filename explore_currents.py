"""
Gulf Stream Surface Current — Exploration & Analysis
=====================================================
Data source: CMEMS GLOBAL_ANALYSISFORECAST_PHY_001_024
             Hourly merged surface currents (model + tidal)
             Depth: ~0.5 m (nearest-surface layer)
             Domain: Gulf of Mexico / Gulf Stream corridor
             Time:   Jan 1 – Apr 17 2026  (2545 hourly snapshots)

Variables
---------
utotal : eastward velocity component  (m/s)
vtotal : northward velocity component (m/s)

Sections
--------
  1–5.  Initial exploration: mean vector map, time series, speed/direction
        distributions, Hovmöller diagram, hourly snapshots.

  6.    Spectrogram of basin-mean speed.
        Reveals which periodicities (tidal, synoptic) are active and when.

  7.    Okubo-Weiss parameter  W = Sn² + Ss² − ζ²
        Classifies every grid point as eddy-interior (W < 0) or
        strain/jet-dominated (W > 0).

  8.    EOF decomposition of the speed anomaly field.
        Finds the dominant independent spatial modes and their time envelopes.
        EOF = PCA applied to a spatiotemporal field.

All figures saved as PNG in the working directory.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import spectrogram
from sklearn.decomposition import PCA

# ── 1. Load data ───────────────────────────────────────────────────────────────
FILE = "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i_1775676050764.nc"
ds = xr.open_dataset(FILE)

# The dataset has a single depth level at 0.494 m. Squeezing it out keeps
# U and V as clean (time, latitude, longitude) arrays.
U = ds["utotal"].squeeze("depth")   # eastward velocity  (time, lat, lon), m/s
V = ds["vtotal"].squeeze("depth")   # northward velocity (time, lat, lon), m/s

lon  = ds["longitude"].values   # (246,)  from -100.67 to -80.25
lat  = ds["latitude"].values    # (154,)  from  19.08 to  31.83
time = U["time"].values         # (2545,) datetime64 at hourly cadence


# ── 2. Derived fields ──────────────────────────────────────────────────────────
# Speed is the Euclidean norm of the two components — this is the scalar
# quantity we will most directly map to pitch or loudness in sonification.
speed = np.sqrt(U**2 + V**2)                  # (time, lat, lon)  m/s

# Oceanographic convention: direction is the bearing the current flows TOWARD,
# measured clockwise from North.  arctan2(V, U) gives the math angle (CCW from
# East), so we convert:  bearing = (90 - math_angle) mod 360
direction_math    = np.degrees(np.arctan2(V, U))   # (time, lat, lon)
direction_bearing = (90 - direction_math) % 360


# ── 3. Time-mean and basin-average ────────────────────────────────────────────
# The time mean reveals the persistent circulation skeleton — Gulf Stream jet,
# Loop Current, etc.  The basin average compresses space into a single scalar
# per time step, giving a coarse but interpretable signal for early sonification.
mean_U     = U.mean(dim="time")
mean_V     = V.mean(dim="time")
mean_speed = speed.mean(dim="time")

ts_u     = U.mean(dim=["latitude", "longitude"])     # (time,)
ts_v     = V.mean(dim=["latitude", "longitude"])
ts_speed = speed.mean(dim=["latitude", "longitude"])


# ── Helper: consistent time-axis formatting ────────────────────────────────────
def format_time_axis(ax, interval="month"):
    if interval == "week":
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="x", color="gray", lw=0.3, ls=":")


# ── Figure 1: Time-mean vector map ────────────────────────────────────────────
# Color = mean speed; arrows = mean flow direction and magnitude.
# We subsample the quiver grid to avoid overplotting (every 4th point ~0.33° step).

QUIVER_STEP = 4

fig, ax = plt.subplots(figsize=(11, 7))

cf = ax.contourf(lon, lat, mean_speed.values, levels=20, cmap="plasma", alpha=0.9)
fig.colorbar(cf, ax=ax, label="Mean speed (m/s)")

q = ax.quiver(
    lon[::QUIVER_STEP], lat[::QUIVER_STEP],
    mean_U.values[::QUIVER_STEP, ::QUIVER_STEP],
    mean_V.values[::QUIVER_STEP, ::QUIVER_STEP],
    scale=8, width=0.003, color="white", alpha=0.75
)
ax.quiverkey(q, 0.92, 1.03, 0.5, "0.5 m/s", labelpos="E", coordinates="axes")

ax.set_title("Time-Mean Surface Current Velocity  (Jan–Apr 2026)", fontsize=13)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_facecolor("#1a1a2e")
fig.tight_layout()
fig.savefig("fig1_mean_vector_map.png", dpi=150)
plt.close(fig)
print("Saved fig1_mean_vector_map.png")


# ── Figure 2: Basin-averaged time series ──────────────────────────────────────
# Three panels: u(t), v(t), speed(t).
# Averaging over the whole spatial domain flattens out mesoscale features but
# preserves basin-scale events (e.g. storm-driven surges, tidal aliasing).

fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

axes[0].plot(time, ts_u,     lw=0.7, color="#4fc3f7")
axes[0].axhline(0, color="white", lw=0.5, ls="--")
axes[0].set_ylabel("u  (m/s)")
axes[0].set_title("Basin-Averaged Eastward Velocity")
axes[0].set_facecolor("#0d0d0d")

axes[1].plot(time, ts_v,     lw=0.7, color="#a5d6a7")
axes[1].axhline(0, color="white", lw=0.5, ls="--")
axes[1].set_ylabel("v  (m/s)")
axes[1].set_title("Basin-Averaged Northward Velocity")
axes[1].set_facecolor("#0d0d0d")

axes[2].plot(time, ts_speed, lw=0.7, color="#ffb74d")
axes[2].set_ylabel("Speed (m/s)")
axes[2].set_title("Basin-Averaged Current Speed  |U|")
axes[2].set_facecolor("#0d0d0d")

for ax in axes:
    format_time_axis(ax)

fig.suptitle("Surface Current Time Series — Basin Average", fontsize=13)
fig.tight_layout()
fig.savefig("fig2_timeseries.png", dpi=150)
plt.close(fig)
print("Saved fig2_timeseries.png")


# ── Figure 3: Speed histogram + direction rose ────────────────────────────────
# The histogram shows the full dynamic range of speeds across all grid points
# and all time steps
#
# The rose plot shows whether the flow has a preferred direction.
# A strong lobe indicates a jet; isotropy suggests turbulent/eddy-dominated flow.

all_speed = speed.values.ravel()
all_speed = all_speed[~np.isnan(all_speed)]

all_bearing = direction_bearing.values.ravel()
all_bearing = all_bearing[~np.isnan(all_bearing)]

fig = plt.figure(figsize=(12, 5))

# --- Speed histogram ---
ax_hist = fig.add_subplot(1, 2, 1)
ax_hist.hist(all_speed, bins=120, color="#ff8a65", edgecolor="none", density=True)
ax_hist.axvline(np.mean(all_speed),
                color="white",  lw=1.2, ls="--",
                label=f"Mean = {np.mean(all_speed):.3f} m/s")
ax_hist.axvline(np.percentile(all_speed, 95),
                color="#ffee58", lw=1.2, ls=":",
                label=f"95th pct = {np.percentile(all_speed, 95):.3f} m/s")
ax_hist.set_xlabel("Speed (m/s)")
ax_hist.set_ylabel("Probability density")
ax_hist.set_title("Speed Distribution\n(all grid points × all time steps)")
ax_hist.legend(fontsize=9)
ax_hist.set_facecolor("#1a1a1a")

# --- Directional rose ---
# 36 bins of 10° each.  Bar height = fraction of total observations in that bin.
N_BINS = 36
bin_edges   = np.linspace(0, 360, N_BINS + 1)
counts, _   = np.histogram(all_bearing, bins=bin_edges)
bin_centers = np.deg2rad(bin_edges[:-1] + 5)  # centre of each 10° wedge

ax_rose = fig.add_subplot(1, 2, 2, projection="polar")
ax_rose.set_theta_zero_location("N")    # 0° at top = North
ax_rose.set_theta_direction(-1)          # clockwise, matching compass convention
ax_rose.bar(bin_centers, counts / counts.sum(),
            width=np.deg2rad(360 / N_BINS),
            color="#80deea", edgecolor="black", linewidth=0.3, alpha=0.85)
ax_rose.set_title("Current Direction Rose\n(bearing toward, all grid × time)",
                  pad=15, fontsize=10)
ax_rose.set_yticklabels([])
ax_rose.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
ax_rose.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

fig.tight_layout()
fig.savefig("fig3_speed_dist_and_rose.png", dpi=150)
plt.close(fig)
print("Saved fig3_speed_dist_and_rose.png")


# ── Figure 4: Hovmöller diagram ───────────────────────────────────────────────
# Collapse longitude (zonal mean) to get speed(time, latitude).
# This shows whether high-speed events are latitude-banded — for example,
# the Gulf Stream core typically sits near 25–28 N in this domain,
# and a persistent bright band there would confirm it.

hov_speed = speed.mean(dim="longitude")   # (time, lat)
vmax_hov  = float(np.nanpercentile(hov_speed.values, 99))

fig, ax = plt.subplots(figsize=(13, 6))
im = ax.pcolormesh(time, lat, hov_speed.values.T,
                   cmap="YlOrRd", shading="auto",
                   vmin=0, vmax=vmax_hov)
fig.colorbar(im, ax=ax, label="Zonal-mean speed (m/s)")
ax.set_ylabel("Latitude (°N)")
ax.set_title("Hovmöller Diagram — Zonal-Mean Speed vs. Latitude  (Jan–Apr 2026)",
             fontsize=12)
format_time_axis(ax)
fig.tight_layout()
fig.savefig("fig4_hovmoller.png", dpi=150)
plt.close(fig)
print("Saved fig4_hovmoller.png")


# ── Figure 5: Four hourly snapshots ───────────────────────────────────────────
# Evenly spaced instants across the record.  Reveals mesoscale structure:
# eddies, filaments, and the Gulf Stream meander that would be invisible in the mean.

snap_indices = np.linspace(0, len(time) - 1, 4, dtype=int)
vmax_snap    = float(np.nanpercentile(speed.values, 98))

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.ravel()

for i, idx in enumerate(snap_indices):
    ax   = axes[i]
    s_i  = speed.isel(time=idx).values
    u_i  = U.isel(time=idx).values
    v_i  = V.isel(time=idx).values

    cf = ax.contourf(lon, lat, s_i, levels=20, cmap="plasma",
                     vmin=0, vmax=vmax_snap)
    ax.quiver(lon[::QUIVER_STEP], lat[::QUIVER_STEP],
              u_i[::QUIVER_STEP, ::QUIVER_STEP],
              v_i[::QUIVER_STEP, ::QUIVER_STEP],
              scale=8, width=0.003, color="white", alpha=0.55)
    ax.set_title(str(time[idx])[:13], fontsize=9)
    ax.set_xlabel("Lon (°E)")
    ax.set_ylabel("Lat (°N)")
    fig.colorbar(cf, ax=ax, label="Speed (m/s)")

fig.suptitle("Hourly Snapshots — Surface Current Speed + Vectors", fontsize=12)
fig.tight_layout()
fig.savefig("fig5_snapshots.png", dpi=150)
plt.close(fig)
print("Saved fig5_snapshots.png")


# ── Console summary ───────────────────────────────────────────────────────────
print("\n── Dataset summary ──────────────────────────────────────────────")
print(f"  Grid:   {len(lat)} × {len(lon)}  (lat × lon),  {len(time)} time steps")
print(f"  Domain: {lat.min():.2f}–{lat.max():.2f} °N,  "
      f"{lon.min():.2f}–{lon.max():.2f} °E")
print(f"  Period: {str(time[0])[:10]}  →  {str(time[-1])[:10]}")
print(f"  u:      {float(U.min()):.3f} to {float(U.max()):.3f} m/s")
print(f"  v:      {float(V.min()):.3f} to {float(V.max()):.3f} m/s")
print(f"  Speed:  {all_speed.min():.3f} to {all_speed.max():.3f} m/s  "
      f"(mean {np.mean(all_speed):.3f}, 95th pct {np.percentile(all_speed, 95):.3f})")


# ── 6. Spectrogram of basin-mean speed ────────────────────────────────────────
# 240-hour (10-day) Hann window and
# 50 % overlap.  This gives ~2-hour frequency resolution and ~5-day time steps —
# fine enough to resolve M2 tidal (12.4 h), K1 diurnal (24 h), and the inertial
# period (~28 h at 25 N), while showing how their energy varies over the record.
#
# Period is on the y-axis (1/frequency)

ts_speed_arr = ts_speed.values.copy()
ts_speed_arr[np.isnan(ts_speed_arr)] = np.nanmean(ts_speed_arr)

WIN_H = 240     # window length in hours
OVER  = WIN_H // 2

f_sg, t_sg, Sxx = spectrogram(ts_speed_arr, fs=1.0, nperseg=WIN_H,
                               noverlap=OVER, window="hann", scaling="density")

# Convert the segment-offset time array (hours from start) to actual timestamps
t_sg_dt = time[0] + t_sg.astype("timedelta64[h]")

# Reference periodicities we expect in tidal + oceanic data
TIDAL_PERIODS = {
    "M2  (12.4 h)":          12.42,
    "K1  (24 h)":            24.00,
    "Inertial (~28 h @ 25N)": 2 * np.pi / (2 * 7.2921e-5 * np.sin(np.deg2rad(25))) / 3600,
}

# Restrict to 5–200 h: avoids the poorly-sampled very-low-frequency tail
freq_mask  = (f_sg > 1 / 200) & (f_sg <= 1 / 5)
periods_sg = 1.0 / f_sg[freq_mask]

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.pcolormesh(
    t_sg_dt, periods_sg, Sxx[freq_mask, :],
    shading="auto", cmap="inferno",
    norm=plt.matplotlib.colors.LogNorm(
        vmin=np.nanpercentile(Sxx[freq_mask], 5),
        vmax=np.nanpercentile(Sxx[freq_mask], 99)
    )
)
fig.colorbar(im, ax=ax, label="PSD  (m²/s² per cyc/hr)")
for label, p in TIDAL_PERIODS.items():
    ax.axhline(p, color="#80deea", lw=0.9, ls="--", alpha=0.85)
    ax.text(t_sg_dt[-1], p, label, color="#80deea",
            fontsize=7.5, va="center", ha="right")
ax.set_xlabel("Date")
ax.set_ylabel("Period (hours)")
ax.set_title("Spectrogram of Basin-Mean Current Speed  —  periods 5–200 h  (log PSD)",
             fontsize=12)
format_time_axis(ax, interval="week")
fig.tight_layout()
fig.savefig("fig6_spectrogram.png", dpi=150)
plt.close(fig)
print("Saved fig6_spectrogram.png")


# ── 7. Okubo-Weiss Parameter ──────────────────────────────────────────────────
# W = Sn² + Ss² − ζ²
#
#   Sn = ∂u/∂x − ∂v/∂y   normal strain    (stretching along the flow axis)
#   Ss = ∂v/∂x + ∂u/∂y   shear strain     (rotational shear)
#   ζ  = ∂v/∂x − ∂u/∂y   relative vorticity
#
# All components are velocity gradients in s⁻¹; W has units s⁻².
# Sign of W tells us the local flow regime:
#   W > 0  →  strain dominates  →  jet or filament (deformation zone)
#   W < 0  →  rotation dominates →  coherent eddy interior
#
# Eddy-core threshold: W < −0.2 σ_W, where σ_W is the spatial standard
# deviation at each time step (Isern-Fontanet et al. 2003).
#
# Grid conversion — the data are on a regular lat-lon grid (~0.0833° spacing).
# Velocity gradients must be in physical units (m/s per m = s⁻¹):
#   dy = R · Δφ_rad                  (constant)
#   dx = R · cos(φ) · Δλ_rad        (latitude-dependent)
#
# np.gradient without a spacing argument uses index spacing = 1.
# Dividing by the physical metres-per-grid-cell converts to s⁻¹.

R_EARTH  = 6.371e6   # m
dlat_deg = float(lat[1] - lat[0])
dlon_deg = float(lon[1] - lon[0])
dy_m     = R_EARTH * np.deg2rad(dlat_deg)                            # scalar, m
dx_m     = R_EARTH * np.cos(np.deg2rad(lat)) * np.deg2rad(dlon_deg)  # (n_lat,), m

u_arr = U.values   # (time, lat, lon)
v_arr = V.values

dudy = np.gradient(u_arr, dy_m, axis=1)                                         # ∂u/∂y
dvdy = np.gradient(v_arr, dy_m, axis=1)                                         # ∂v/∂y
dudx = np.gradient(u_arr, axis=2) / dx_m[np.newaxis, :, np.newaxis]             # ∂u/∂x
dvdx = np.gradient(v_arr, axis=2) / dx_m[np.newaxis, :, np.newaxis]             # ∂v/∂x

Sn   = dudx - dvdy    # normal strain
Ss   = dvdx + dudy    # shear strain
zeta = dvdx - dudy    # relative vorticity  (positive = cyclonic / CCW in NH)
W    = Sn**2 + Ss**2 - zeta**2   # (time, lat, lon)  s⁻²

# Time-mean W field
W_mean   = np.nanmean(W, axis=0)
W_abs_99 = np.nanpercentile(np.abs(W_mean), 99)

fig, ax = plt.subplots(figsize=(11, 7))
im = ax.pcolormesh(lon, lat, W_mean, cmap="RdBu_r", shading="auto",
                   vmin=-W_abs_99, vmax=W_abs_99)
fig.colorbar(im, ax=ax, label="W  (s⁻²)")
ax.set_title("Time-Mean Okubo-Weiss Parameter  (Jan–Apr 2026)\n"
             "Blue = rotation-dominated (eddies)     Red = strain-dominated (jets/filaments)",
             fontsize=11)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
fig.tight_layout()
fig.savefig("fig7_ow_mean.png", dpi=150)
plt.close(fig)
print("Saved fig7_ow_mean.png")

# Three snapshots of W with eddy-core contours
snap_indices_ow = np.linspace(0, len(time) - 1, 3, dtype=int)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, idx in enumerate(snap_indices_ow):
    ax       = axes[i]
    W_snap   = W[idx]
    sigma_W  = np.nanstd(W_snap)
    eddy_core = W_snap < -0.2 * sigma_W
    W_99     = np.nanpercentile(np.abs(W_snap), 99)
    im       = ax.pcolormesh(lon, lat, W_snap, cmap="RdBu_r", shading="auto",
                             vmin=-W_99, vmax=W_99)
    ax.contour(lon, lat, eddy_core.astype(float), levels=[0.5],
               colors="white", linewidths=0.8)
    ax.set_title(str(time[idx])[:13], fontsize=9)
    ax.set_xlabel("Lon (°E)")
    ax.set_ylabel("Lat (°N)")
    fig.colorbar(im, ax=ax, label="W (s⁻²)")
fig.suptitle("Okubo-Weiss Snapshots  —  White contour = eddy cores  (W < −0.2 σ_W)",
             fontsize=11)
fig.tight_layout()
fig.savefig("fig8_ow_snapshots.png", dpi=150)
plt.close(fig)
print("Saved fig8_ow_snapshots.png")

# Eddy-core fraction time series — what fraction of the domain is inside an eddy?
sigma_W_t     = np.nanstd(W, axis=(1, 2))
eddy_fraction = np.nanmean(W < -0.2 * sigma_W_t[:, np.newaxis, np.newaxis], axis=(1, 2))

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(time, eddy_fraction * 100, lw=0.7, color="#ce93d8")
ax.set_ylabel("% of domain in eddy cores")
ax.set_title("Fraction of Domain Classified as Eddy Interior  (W < −0.2 σ_W)", fontsize=11)
ax.set_facecolor("#0d0d0d")
format_time_axis(ax)
fig.tight_layout()
fig.savefig("fig9_eddy_fraction.png", dpi=150)
plt.close(fig)
print("Saved fig9_eddy_fraction.png")


# ── 8. EOF Decomposition ──────────────────────────────────────────────────────
#   Reshape speed(t, φ, λ) from (N_t, N_lat, N_lon) to a 2D matrix
#   X of shape (N_t, N_space).  Each row is one hourly map flattened to a vector.
#
#   Remove the time mean at each spatial point so we analyse anomalies
#   (deviations from the mean circulation), not the mean itself:
#     X_anom = X − mean(X, axis=0)
#
# The decomposition via SVD:
#   X_anom = U Σ Vᵀ
#   EOF_k  = V_k  — k-th right singular vector, shape (N_space,)
#              reshaped to (lat, lon): the k-th spatial pattern
#   PC_k   = U_k Σ_k — temporal amplitude of pattern k, shape (N_t,)
#   Variance explained by mode k = Σ_k² / Σ_i Σ_i²
#
# Physical interpretation for this domain:
#   EOF1  likely captures jet strengthening/weakening: positive loading along
#         the Gulf Stream axis, PC1 is large when the jet is strong.
#   EOF2  likely captures meridional meanders: alternating sign north/south
#         of the jet core.
#   Higher modes capture eddy activity, Loop Current pulses, etc.
#
# Why this matters for sonification:
#   Each EOF mode provides:
#     - A spatial pattern → which region of the domain is "active"
#       (could drive spatial audio or timbral color per region)
#     - A temporal envelope PC_k(t) that is uncorrelated with all other modes
#       → each audio layer is statistically independent

N_t, N_lat, N_lon = len(time), len(lat), len(lon)
speed_2d  = speed.values.reshape(N_t, N_lat * N_lon)
land_mask = np.all(np.isnan(speed_2d), axis=0)
speed_2d  = np.where(np.isnan(speed_2d), 0.0, speed_2d)  # fill land with 0
speed_anom = speed_2d - speed_2d.mean(axis=0)

N_MODES = 10
pca  = PCA(n_components=N_MODES)
PCs  = pca.fit_transform(speed_anom)              # (N_t, N_MODES) temporal envelopes
EOFs = pca.components_.reshape(N_MODES, N_lat, N_lon)   # spatial patterns
var_explained = pca.explained_variance_ratio_

# Mask land back to NaN
for k in range(N_MODES):
    EOFs[k][land_mask.reshape(N_lat, N_lon)] = np.nan

# --- Variance explained bar chart ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(np.arange(1, N_MODES + 1), var_explained * 100,
       color="#4fc3f7", edgecolor="black")
ax.plot(np.arange(1, N_MODES + 1), np.cumsum(var_explained) * 100,
        color="#ff8a65", marker="o", ms=4, lw=1.2, label="Cumulative")
ax.set_xlabel("EOF mode")
ax.set_ylabel("Variance explained (%)")
ax.set_title("EOF Decomposition — Variance Explained by Each Mode")
ax.legend()
ax.set_facecolor("#0d0d0d")
fig.tight_layout()
fig.savefig("fig10_eof_variance.png", dpi=150)
plt.close(fig)
print("Saved fig10_eof_variance.png")

# --- Leading 4 EOF spatial patterns ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for k, ax in enumerate(axes.ravel()):
    vmax = np.nanpercentile(np.abs(EOFs[k]), 99)
    im   = ax.pcolormesh(lon, lat, EOFs[k], cmap="RdBu_r", shading="auto",
                         vmin=-vmax, vmax=vmax)
    ax.set_title(f"EOF {k+1}  —  {var_explained[k]*100:.1f}% of variance", fontsize=10)
    ax.set_xlabel("Lon (°E)")
    ax.set_ylabel("Lat (°N)")
    fig.colorbar(im, ax=ax, label="Loading")
fig.suptitle("Leading EOF Spatial Patterns of Speed Anomaly", fontsize=13)
fig.tight_layout()
fig.savefig("fig11_eof_patterns.png", dpi=150)
plt.close(fig)
print("Saved fig11_eof_patterns.png")

# --- PC time series (leading 4 modes) ---
fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
colors_pc = ["#4fc3f7", "#a5d6a7", "#ff8a65", "#ce93d8"]
for k, ax in enumerate(axes):
    ax.plot(time, PCs[:, k], lw=0.6, color=colors_pc[k])
    ax.axhline(0, color="white", lw=0.4, ls="--")
    ax.set_ylabel(f"PC {k+1}")
    ax.set_title(f"PC {k+1}  ({var_explained[k]*100:.1f}% variance)", fontsize=9)
    ax.set_facecolor("#0d0d0d")
    format_time_axis(ax)
fig.suptitle("Principal Component Time Series — Temporal Envelopes of Each EOF Mode",
             fontsize=12)
fig.tight_layout()
fig.savefig("fig12_eof_pcs.png", dpi=150)
plt.close(fig)
print("Saved fig12_eof_pcs.png")

# --- Spectrogram of PC1 — what frequencies drive the dominant mode? ---
# If EOF1 captures jet variability, this tells us whether it is tidal,
# synoptic (~3–7 days), or lower-frequency (weeks).
pc1 = PCs[:, 0] - PCs[:, 0].mean()
f_pc, t_pc, Sxx_pc = spectrogram(pc1, fs=1.0, nperseg=WIN_H, noverlap=OVER,
                                  window="hann", scaling="density")
t_pc_dt   = time[0] + t_pc.astype("timedelta64[h]")
freq_mask2 = (f_pc > 1 / 200) & (f_pc <= 1 / 5)
periods2   = 1.0 / f_pc[freq_mask2]

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.pcolormesh(
    t_pc_dt, periods2, Sxx_pc[freq_mask2, :],
    shading="auto", cmap="inferno",
    norm=plt.matplotlib.colors.LogNorm(
        vmin=np.nanpercentile(Sxx_pc[freq_mask2], 5),
        vmax=np.nanpercentile(Sxx_pc[freq_mask2], 99)
    )
)
fig.colorbar(im, ax=ax, label="PSD (arbitrary units)")
for label, p in TIDAL_PERIODS.items():
    ax.axhline(p, color="#80deea", lw=0.9, ls="--", alpha=0.85)
    ax.text(t_pc_dt[-1], p, label, color="#80deea",
            fontsize=7.5, va="center", ha="right")
ax.set_xlabel("Date")
ax.set_ylabel("Period (hours)")
ax.set_title("Spectrogram of PC1 — Temporal Frequencies of the Dominant EOF Mode",
             fontsize=12)
format_time_axis(ax, interval="week")
fig.tight_layout()
fig.savefig("fig13_eof_pc1_spectrogram.png", dpi=150)
plt.close(fig)
print("Saved fig13_eof_pc1_spectrogram.png")

# ── Final console summary ─────────────────────────────────────────────────────
print(f"\nEOF variance explained (modes 1–{N_MODES}):")
cumvar = 0.0
for k in range(N_MODES):
    cumvar += var_explained[k] * 100
    print(f"  Mode {k+1:2d}: {var_explained[k]*100:5.2f}%   cumulative {cumvar:5.2f}%")
