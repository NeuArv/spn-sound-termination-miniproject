"""Model version of Kopp-Scheinpflug et al. (2011) Fig. 1C.

A 200 ms, -400 pA hyperpolarizing current step causes a hyperpolarization
with a "sag" (I_H activation) and, after the step, a rapid return to rest with
an offset burst of APs. Inset: ten superimposed trials (with conductance
noise, as in the paper's methods) at higher gain show the well-timed burst.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from neuron import h

os.chdir(os.path.dirname(os.path.abspath(__file__)))
h.load_file('stdrun.hoc')
h.load_file('spn_neuron.hoc')
seg = h.soma(0.5)

AMP = -0.4            # nA
DUR = 200.0           # ms
T_ON = 300.0          # ms, settle at rest first
T_PRE, T_POST = 30.0, 80.0   # ms shown before onset / after offset
N_TRIALS = 10

stim = h.IClamp(seg)
stim.delay, stim.dur, stim.amp = T_ON, DUR, AMP

# excitatory conductance noise (paper methods: mean 1 pS, SD 0.5 nS, tau 2 ms)
fl = h.mhh_Gfluct(seg)
fl.g_e0, fl.std_e, fl.tau_e, fl.E_e = 1e-6, 0.0005, 2, 0

t_vec = h.Vector().record(h._ref_t)
v_vec = h.Vector().record(seg._ref_v)
i_vec = h.Vector().record(stim._ref_i)
rng = h.NetStims(0.5)
rng.seed(3)           # seeds NEURON's random generator -> reproducible noise

trials = []
for _ in range(N_TRIALS):
    h.dt = 0.025
    h.finitialize(-60)
    h.continuerun(T_ON + DUR + T_POST)
    t = np.array(t_vec) - T_ON
    trials.append((t, np.array(v_vec), np.array(i_vec)))

t, v, i = trials[0]
keep = t >= -T_PRE
v_rest = v[(t > -T_PRE) & (t < 0)].mean()

fig = plt.figure(figsize=(6.5, 4.2))
ax_v = fig.add_axes([0.14, 0.30, 0.50, 0.62])
ax_i = fig.add_axes([0.14, 0.08, 0.50, 0.12], sharex=ax_v)
ax_in = fig.add_axes([0.68, 0.45, 0.31, 0.50])

# main voltage trace
ax_v.plot(t[keep], v[keep], 'k', lw=0.8)
ax_v.text(-T_PRE - 4, v_rest, f'{v_rest:.0f} mV', ha='right', va='center',
          fontsize=10)
ax_v.set_ylim(-110, 45)

# current step (lower trace)
ax_i.plot(t[keep], i[keep] * 1e3, 'k', lw=1)
ax_i.set_ylim(AMP * 1e3 * 1.3, 60)
ax_i.text(DUR / 2, AMP * 1e3 * 1.25, f'{DUR:.0f} ms', ha='center',
          va='top', fontsize=10)
ax_i.text(-T_PRE - 4, AMP * 1e3 / 2, f'{AMP*1e3:.0f} pA', ha='right',
          va='center', fontsize=9)

# inset: offset bursts of all trials at higher gain
for tt, vv, _ in trials:
    w = (tt >= DUR - 2) & (tt <= DUR + 25)
    ax_in.plot(tt[w] - DUR, vv[w], color='0.15', lw=0.6, alpha=0.8)
ax_in.set_xlim(-2, 38)
ax_in.set_ylim(-90, 45)
# scale bars: 40 mV, 10 ms (placed right of the traces)
x0, y0 = 37, -85
ax_in.plot([x0 - 10, x0], [y0, y0], 'k', lw=1.5, clip_on=False)
ax_in.plot([x0, x0], [y0, y0 + 40], 'k', lw=1.5, clip_on=False)
ax_in.text(x0 + 1, y0 + 20, '40 mV', ha='left', va='center', fontsize=9)
ax_in.text(x0 - 5, y0 - 3, '10 ms', ha='center', va='top', fontsize=9)

for ax in (ax_v, ax_i, ax_in):
    ax.axis('off')
ax_v.set_xlim(-T_PRE, DUR + T_POST)
fig.text(0.01, 0.95, 'C', fontsize=16, fontweight='bold')
fig.savefig('fig1C_model.png', dpi=300, bbox_inches='tight')

sp = [int(np.sum((vv[1:] > 0) & (vv[:-1] <= 0) & (tt[1:] > DUR)))
      for tt, vv, _ in trials]
print(f'rest {v_rest:.1f} mV, min {v.min():.1f} mV, '
      f'end of step {v[np.searchsorted(t, DUR - 0.5)]:.1f} mV')
print(f'offset APs per trial: {sp}')
print('saved fig1C_model.png')
