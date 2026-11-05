"""Gap detection in the SPN model (cf. Kopp-Scheinpflug et al. 2011, Fig. 8).

Same synaptic protocol as spn_model.ipynb: a train of IPSPs at 100 Hz
(14 synapses x 4 nS, E_rev = -100 mV) mimics the MNTB input during a sound.
The train is interrupted by a silent gap, then a second train starts.
A gap is "detected" if the SPN fires an offset AP in the gap.

  (A) 200 ms pregap, gaps of 50, 3 and 1 ms
  (B) 3 ms gap after pregap durations of 300, 100 and 50 ms
  (C) detection probability vs. gap length and pregap duration (noise trials)
  (D) gap threshold vs. pregap duration: control, I_H -50 %, E_Cl +20 mV
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from neuron import h

os.chdir(os.path.dirname(os.path.abspath(__file__)))
h.load_file('stdrun.hoc')
h.load_file('spn_neuron.hoc')
seg = h.soma(0.5)
GH0 = seg.sjg_ih.ghbar

NSYNS = 14            # inhibitory synapses
G_SYN = 4e-3          # uS (4 nS) per synapse
E_INH = -100.0        # mV, inhibitory reversal potential (E_Cl)
ISI = 10.0            # ms, 100 Hz IPSP train
T_END = 300.0         # ms, when the next IPSP of the pregap train would come
POST = 30.0           # ms of second train to simulate after the gap
N_TRIALS = 20         # noise trials per point in (C) and (D)
PREGAPS = np.arange(50, 301, 25)
GAPS = np.array([0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20])

# one Exp2Syn with summed weight == 14 identical synapses driven together
syn = h.Exp2Syn(seg)
syn.tau1, syn.tau2 = 0.1, 2.0
train1, train2 = h.NetStims(0.5), h.NetStims(0.5)
nc1, nc2 = h.NetCon(train1, syn), h.NetCon(train2, syn)
nc1.weight[0] = nc2.weight[0] = NSYNS * G_SYN

# excitatory conductance noise, as in the notebook
fl = h.mhh_Gfluct(seg)
fl.g_e0, fl.std_e, fl.tau_e, fl.E_e = 1e-6, 0.0005, 2, 0

t_vec = h.Vector().record(h._ref_t)
v_vec = h.Vector().record(seg._ref_v)
spikes = h.Vector()
det = h.NetCon(seg._ref_v, None, sec=h.soma)
det.threshold = -10
det.record(spikes)
train1.seed(1)        # seeds NEURON's random generator -> reproducible noise


def trial(pregap, gap, post=POST, e_inh=E_INH, gh=GH0):
    """One run. Returns t, v (time 0 = end of pregap train) and gap-AP flag."""
    n = int(round(pregap / ISI))
    train1.start, train1.interval, train1.number = T_END - ISI * n, ISI, n
    train2.start, train2.interval = T_END + gap, ISI
    train2.number = int(post / ISI) + 1
    syn.e = e_inh
    seg.sjg_ih.ghbar = gh
    h.dt = 0.025
    h.finitialize(-60)
    h.continuerun(T_END + gap + post)
    sp = np.array(spikes)
    # offset AP: after the last pregap IPSP, before the 2nd train can stop it
    hit = np.any((sp > T_END - ISI) & (sp <= T_END + gap + 5))
    return np.array(t_vec) - T_END, np.array(v_vec), hit


def gap_threshold(prob):
    """Smallest gap with >= 50 % detection (linear interpolation)."""
    idx = np.where(prob >= 0.5)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    if i == 0:
        return GAPS[0]
    g0, g1, p0, p1 = GAPS[i - 1], GAPS[i], prob[i - 1], prob[i]
    return g0 + (0.5 - p0) * (g1 - g0) / (p1 - p0)


def sweep(**kw):
    prob = np.zeros((len(PREGAPS), len(GAPS)))
    for i, pre in enumerate(PREGAPS):
        for j, gap in enumerate(GAPS):
            prob[i, j] = np.mean([trial(pre, gap, **kw)[2]
                                  for _ in range(N_TRIALS)])
        print(f'  pregap {pre:3.0f} ms done')
    return prob


def plot_traces(ax, runs, title):
    """Stack traces vertically; orange = silent gap (incl. normal ISI)."""
    for k, (pre, gap, label) in enumerate(runs):
        t, v, hit = trial(pre, gap, post=60)
        off = -k * 130
        ax.fill_between([-ISI, gap], off - 100, off + 40, color='orange',
                        alpha=0.3, lw=0)
        ax.plot(t, v + off, 'k', lw=0.7)
        ax.text(-ISI - 5, off + 30, label + ('  -> AP' if hit else
                                              '  -> no AP'),
                ha='right', fontsize=9)
    ax.set_xlim(-120, 70)
    ax.set_yticks([])
    ax.set_xlabel('Time from end of pregap train (ms)')
    ax.set_title(title, loc='left', fontsize=11)
    ax.spines[['top', 'right', 'left']].set_visible(False)


fig, axs = plt.subplots(2, 2, figsize=(13, 10))
plot_traces(axs[0, 0], [(200, 50, '50 ms gap'), (200, 3, '3 ms gap'),
                        (200, 1, '1 ms gap')],
            '(A) 200 ms pregap, different gaps')
plot_traces(axs[0, 1], [(300, 3, '300 ms pregap'), (100, 3, '100 ms pregap'),
                        (50, 3, '50 ms pregap')],
            '(B) 3 ms gap, different pregap durations')

print('control sweep')
prob = sweep()
thr = {'Control': np.array([gap_threshold(p) for p in prob])}
print('I_H -50 % sweep')
thr['$I_H$ -50 %'] = np.array([gap_threshold(p)
                               for p in sweep(gh=0.5 * GH0)])
print('E_Cl +20 mV sweep')
thr['$E_{Cl}$ +20 mV'] = np.array([gap_threshold(p)
                                   for p in sweep(e_inh=E_INH + 20)])
seg.sjg_ih.ghbar, syn.e = GH0, E_INH

ax = axs[1, 0]
im = ax.pcolormesh(PREGAPS, np.arange(len(GAPS)), prob.T, cmap='Greys',
                   vmin=0, vmax=1, shading='nearest')
ax.plot(PREGAPS, np.interp(thr['Control'], GAPS, np.arange(len(GAPS))),
        'r-o', ms=4, label='threshold (50 %)')
ax.set_yticks(np.arange(len(GAPS)), [f'{g:g}' for g in GAPS])
ax.set_xlabel('Pregap duration (ms)')
ax.set_ylabel('Gap length (ms)')
ax.set_title(f'(C) Detection probability ({N_TRIALS} noise trials)',
             loc='left', fontsize=11)
ax.legend(frameon=False, loc='upper right')
fig.colorbar(im, ax=ax, label='P(offset AP in gap)')

ax = axs[1, 1]
for (name, th), col in zip(thr.items(), ['k', 'green', 'tab:blue']):
    ax.plot(PREGAPS, th, 'o-', color=col, label=name)
ax.set_xlabel('Pregap duration (ms)')
ax.set_ylabel('Gap threshold (ms)')
ax.set_ylim(0, None)
ax.set_title('(D) Gap threshold', loc='left', fontsize=11)
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)

for name, th in thr.items():
    print(f'{name.replace("$", ""):16s} threshold (ms): ' +
          ' '.join('  -- ' if np.isnan(x) else f'{x:5.1f}' for x in th))
fig.tight_layout()
fig.savefig('gap_detection.png', dpi=150, bbox_inches='tight')
print('saved gap_detection.png')
