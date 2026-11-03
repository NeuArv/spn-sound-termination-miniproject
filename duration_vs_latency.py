"""Recreate Kopp-Scheinpflug et al. (2011) Neuron, Figure 7F/G (model part).

A -400 pA hyperpolarizing current step of 10-100 ms is injected into the SPN
model; the latency of the first offset AP (measured from step offset) is
plotted against step duration for four model conditions:
  full model, no I_H, no I_H with Vm correction, no I_TCa.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from neuron import h

os.chdir(os.path.dirname(os.path.abspath(__file__)))
h.load_file('stdrun.hoc')
h.load_file('spn_neuron.hoc')
soma = h.soma

AMP = -0.4            # nA (-400 pA)
T_ON = 200.0          # ms, step onset (allows settling to rest)
T_POST = 100.0        # ms, window after step offset to look for APs
AP_THRESH = 0.0       # mV, spike detection threshold
DURATIONS = np.arange(10, 101, 10)
GH0, GCA0 = soma(0.5).sjg_ih.ghbar, soma(0.5).lva.gbar

stim = h.IClamp(soma(0.5))
stim.delay = T_ON
# holding current so every condition starts from the full model's resting Vm
hold = h.IClamp(soma(0.5))
hold.delay, hold.dur = 0, 1e9
t_vec, v_vec = h.Vector().record(h._ref_t), h.Vector().record(soma(0.5)._ref_v)
spikes = h.Vector()
nc = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
nc.threshold = AP_THRESH
nc.record(spikes)


def set_condition(gh, gca):
    for seg in soma:
        seg.sjg_ih.ghbar = gh
        seg.lva.gbar = gca


def run(dur, amp):
    stim.dur, stim.amp = dur, amp
    h.dt = 0.01
    h.finitialize(-60)
    h.continuerun(T_ON + dur + T_POST)
    t, v = np.array(t_vec), np.array(v_vec)
    off = T_ON + dur
    sp = np.array(spikes)
    sp = sp[sp > off] - off
    return t - off, v, sp


def steady_state_v(amp, dur=100.0):
    t, v, _ = run(dur, amp)
    return v[np.searchsorted(t, -0.5)]  # Vm just before step offset


def resting_v():
    t, v, _ = run(1.0, 0.0)
    return v[np.searchsorted(t, -2.0)]


def bisect(f, target, lo, hi, n=40):
    """Find x in [lo, hi] with f(x) == target, assuming f increasing in x."""
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)



conditions = {
    'Full model':            dict(gh=GH0, gca=GCA0, color='k'),
    'No $I_H$':              dict(gh=0.0, gca=GCA0, color='green'),
    'No $I_H$, $V_m$ corr.': dict(gh=0.0, gca=GCA0, color='limegreen'),
    'No $I_{TCa}$':          dict(gh=GH0, gca=0.0, color='deepskyblue'),
}

set_condition(GH0, GCA0)
hold.amp = 0.0
v_rest = resting_v()
v_target = steady_state_v(AMP)

# Holding current per condition to restore the full model's resting Vm.
for name, c in conditions.items():
    set_condition(c['gh'], c['gca'])
    hold.amp = 0.0
    c['hold'] = bisect(lambda x: (setattr(hold, 'amp', x), resting_v())[1],
                       v_rest, -0.1, 0.12)

# Vm correction: reduce the step in the no-I_H model so that the steady-state
# hyperpolarization at the end of a 100 ms step matches the full model.
c = conditions['No $I_H$, $V_m$ corr.']
set_condition(c['gh'], c['gca'])
hold.amp = c['hold']
amp_corr = bisect(steady_state_v, v_target, AMP, 0.0)
print(f'Rest: {v_rest:.1f} mV; full model Vm at end of 100 ms step: '
      f'{v_target:.1f} mV; no-I_H corrected step: {amp_corr*1e3:.0f} pA')
for name, c in conditions.items():
    print(f'  holding current {name:24s}: {c["hold"]*1e3:7.1f} pA')

results, traces = {}, {}
for name, c in conditions.items():
    set_condition(c['gh'], c['gca'])
    hold.amp = c['hold']
    amp = amp_corr if 'corr' in name else AMP
    lat, nap, tr = [], [], []
    for d in DURATIONS:
        t, v, sp = run(d, amp)
        lat.append(sp[0] if len(sp) else np.nan)
        nap.append(len(sp))
        tr.append((t, v))
    results[name] = (np.array(lat), np.array(nap))
    traces[name] = tr
    print(f'{name:28s} latency (ms): ' +
          ' '.join('  -- ' if np.isnan(x) else f'{x:5.1f}' for x in lat) +
          f'   #APs: {nap}')
set_condition(GH0, GCA0)

# ---- plotting ----
fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.35, wspace=0.3)
cmap = plt.get_cmap('jet')
for i, name in enumerate(conditions):
    ax = fig.add_subplot(gs[0, i])
    for j, (t, v) in enumerate(traces[name]):
        ax.plot(t, v, color=cmap(j / (len(DURATIONS) - 1)), lw=0.8)
    ax.axhline(v_target, ls='--', color='k', lw=0.8)
    ax.set_xlim(-130, 70)
    vmin = min(v.min() for _, v in traces[name])
    ax.set_ylim(min(-130, np.floor(vmin / 10) * 10 - 10), 50)
    ax.set_title(f'({"i ii iii iv".split()[i]}) {name}', fontsize=10)
    ax.set_xlabel('Time from step offset (ms)')
    if i == 0:
        ax.set_ylabel('Vm (mV)')
    ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[1, :2])
for name, c in conditions.items():
    lat = results[name][0]
    ax.plot(DURATIONS, lat, 'o-', color=c['color'], label=name)
ax.set_xlim(0, 105)
ax.set_ylim(0, None)
ax.set_xlabel('Stimulus duration (ms)')
ax.set_ylabel('First offset AP latency (ms)')
ax.set_title('Stimulus duration vs. offset AP latency (cf. Fig. 7G)')
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)

ax = fig.add_subplot(gs[1, 2:])
for name, c in conditions.items():
    ax.plot(DURATIONS, results[name][1], 'o-', color=c['color'], label=name)
ax.set_xlabel('Stimulus duration (ms)')
ax.set_ylabel('Number of offset APs')
ax.set_title('AP number (cf. Fig. 7H)')
ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('SPN model: -400 pA hyperpolarizing steps, 10-100 ms')
fig.savefig('duration_vs_latency.png', dpi=150, bbox_inches='tight')
print('saved duration_vs_latency.png')
