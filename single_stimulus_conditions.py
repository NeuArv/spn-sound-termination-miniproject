"""Single-stimulus comparison of SPN model conditions (cf. Kopp-Scheinpflug
et al. 2011, Fig. 7F).

One hyperpolarizing current step (100 ms) is applied to four model variants:
  full model, I_H blocked, I_TCa blocked, I_H + I_TCa blocked.
Each trace is annotated with resting Vm, peak hyperpolarization, steady-state
Vm, sag, first offset AP latency and the number of offset APs.

I_H-blocked variants get a constant holding current that restores the full
model's resting Vm and a reduced step so that Vm settles at -100 mV
(as in paper Fig. 7Fii); with -400 pA only the leak is left and Vm would reach
about -230 mV.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from neuron import h

os.chdir(os.path.dirname(os.path.abspath(__file__)))
h.load_file('stdrun.hoc')
h.load_file('spn_neuron.hoc')
seg = h.soma(0.5)

DUR = 100.0           # ms, step duration
AMP = -0.4            # nA, step for conditions with I_H
V_NOIH = -100.0       # mV, target steady-state Vm when I_H is blocked
T_ON = 200.0          # ms, step onset
T_POST = 150.0        # ms, recorded after step offset
T_OFF = T_ON + DUR
AP_THRESH = 0.0       # mV, spike detection
GH0, GCA0 = seg.sjg_ih.ghbar, seg.lva.gbar

stim = h.IClamp(seg)
stim.delay, stim.dur = T_ON, DUR
hold = h.IClamp(seg)
hold.delay, hold.dur = 0, 1e9
t_vec = h.Vector().record(h._ref_t)
v_vec = h.Vector().record(seg._ref_v)
spikes = h.Vector()
nc = h.NetCon(seg._ref_v, None, sec=h.soma)
nc.threshold = AP_THRESH
nc.record(spikes)


def run(amp, hold_amp, tstop=T_OFF + T_POST):
    stim.amp, hold.amp = amp, hold_amp
    h.dt = 0.01
    h.finitialize(-60)
    h.continuerun(tstop)
    return np.array(t_vec), np.array(v_vec), np.array(spikes)


def v_at(t, v, time):
    return v[np.searchsorted(t, time)]


def bisect(f, target, lo, hi, n=40):
    """Find x in [lo, hi] with f(x) == target, assuming f increasing in x."""
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def set_condition(gh, gca):
    seg.sjg_ih.ghbar = gh
    seg.lva.gbar = gca


conditions = {
    'Full model':              dict(gh=GH0, gca=GCA0, color='k'),
    '$I_H$ blocked':           dict(gh=0.0, gca=GCA0, color='green'),
    '$I_{TCa}$ blocked':       dict(gh=GH0, gca=0.0, color='deepskyblue'),
    '$I_H$ + $I_{TCa}$ blocked': dict(gh=0.0, gca=0.0, color='darkorange'),
}

set_condition(GH0, GCA0)
t, v, _ = run(0.0, 0.0, T_ON)
v_rest = v[-1]

# holding current per condition: restore the full model's resting Vm
# (upper bound keeps the cell below the onset of repetitive firing)
for name, c in conditions.items():
    set_condition(c['gh'], c['gca'])
    c['hold'] = bisect(lambda x: run(0.0, x, T_ON)[1][-1], v_rest, -0.1, 0.12)
set_condition(GH0, GCA0)


def measure(t, v, sp):
    """Extract rest, peak, steady state, sag, latency, AP threshold, #APs."""
    during = (t >= T_ON) & (t <= T_OFF)
    i_min = np.argmin(np.where(during, v, np.inf))
    sp = sp[sp > T_OFF]
    m = dict(
        rest=v_at(t, v, T_ON - 1),
        vmin=v[i_min], t_min=t[i_min] - T_ON,
        vss=v_at(t, v, T_OFF - 0.5),
        n_ap=len(sp),
        latency=sp[0] - T_OFF if len(sp) else np.nan,
        isi=np.diff(sp) if len(sp) > 1 else np.array([]),
    )
    m['sag'] = m['vss'] - m['vmin']
    m['rebound'] = v[t > T_OFF].max()
    if len(sp):
        # AP threshold: Vm where dV/dt first exceeds 20 mV/ms in the 2 ms
        # before the 1st AP (excludes the fast repolarization at step offset)
        dvdt = np.gradient(v, t)
        k = np.where((t > sp[0] - 2) & (t < sp[0]) & (dvdt > 20))[0][0]
        m['ap_thr'] = v[k]
    else:
        m['ap_thr'] = np.nan
    return m


def simulate(match_v):
    """Run every condition once. If match_v, I_H-blocked conditions get a
    reduced step so Vm settles at V_NOIH; otherwise all get AMP."""
    res = {}
    for name, c in conditions.items():
        set_condition(c['gh'], c['gca'])
        if match_v and c['gh'] == 0:
            amp = bisect(lambda x: v_at(*run(x, c['hold'], T_OFF)[:2],
                                        T_OFF - 0.5), V_NOIH, AMP, 0.0)
        else:
            amp = AMP
        t, v, sp = run(amp, c['hold'])
        res[name] = dict(c, amp=amp, t=t - T_ON, v=v, m=measure(t, v, sp))
    set_condition(GH0, GCA0)
    return res


def print_table(res, title):
    hdr = (f'{"condition":26s}{"hold":>7s}{"step":>7s}{"rest":>7s}'
           f'{"peak":>7s}{"t_pk":>6s}{"Vss":>7s}{"sag":>6s}{"lat":>7s}'
           f'{"thr":>7s}{"#AP":>5s}')
    print(f'\n{title}\n{hdr}\n' + '-' * len(hdr))
    for name, c in res.items():
        m = c['m']
        print(f'{name.replace("$", ""):26s}{c["hold"]*1e3:7.0f}'
              f'{c["amp"]*1e3:7.0f}{m["rest"]:7.1f}{m["vmin"]:7.1f}'
              f'{m["t_min"]:6.1f}{m["vss"]:7.1f}{m["sag"]:6.1f}'
              f'{m["latency"]:7.1f}{m["ap_thr"]:7.1f}{m["n_ap"]:5d}')
    print('units: pA (hold, step), mV (Vm values), ms (t_pk = time of peak '
          'hyperpolarization after step onset, lat = first AP after step '
          'offset)')


def plot(res, fname, suptitle='', annotate=True):
    """Plot traces; annotate=False gives a clean version (traces, step
    shading, axes and panel titles only) for slides."""
    # common y-range for all panels so depths are directly comparable
    vlow = min(c['v'].min() for c in res.values())
    ybot = min(-110, np.floor(vlow / 10) * 10 - 10)

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.45,
                          wspace=0.2)
    for i, (name, c) in enumerate(res.items()):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        m = c['m']
        ax.axvspan(0, DUR, color='0.92', zorder=0)
        ax.plot(c['t'], c['v'], color=c['color'], lw=1)
        ax.set_xlim(-50, DUR + T_POST)
        ax.set_ylim(ybot, 50)
        ax.set_title(f'({"abcd"[i]}) {name}', loc='left', fontsize=11)
        ax.set_xlabel('Time from step onset (ms)')
        ax.set_ylabel('Vm (mV)')
        ax.spines[['top', 'right']].set_visible(False)
        if not annotate:
            continue

        ax.text(DUR / 2, 42, f'step {c["amp"]*1e3:.0f} pA', ha='center',
                fontsize=8, color='0.4')
        ax.axhline(m['rest'], ls=':', color='0.5', lw=0.8)
        ax.text(-48, m['rest'] + 2, f'rest {m["rest"]:.1f} mV', fontsize=8,
                color='0.4')

        # peak hyperpolarization and steady state
        ax.plot(DUR - 0.5, m['vss'], 's', color='purple', ms=5)
        if m['sag'] > 1:
            ax.plot(m['t_min'], m['vmin'], 'v', color='crimson', ms=6)
            ax.annotate(f'peak {m["vmin"]:.1f} mV', (m['t_min'], m['vmin']),
                        xytext=(m['t_min'] + 6, m['vmin'] - 9), fontsize=8,
                        color='crimson')
            ax.annotate(f'steady state {m["vss"]:.1f} mV',
                        (DUR - 0.5, m['vss']),
                        xytext=(DUR - 80, m['vss'] + 6), fontsize=8,
                        color='purple')
            ax.annotate('', (DUR - 2, m['vss']), (DUR - 2, m['vmin']),
                        arrowprops=dict(arrowstyle='<->', color='purple',
                                        lw=0.8))
            ax.text(DUR - 4, (m['vss'] + m['vmin']) / 2 - 3,
                    f'sag {m["sag"]:.1f} mV', fontsize=8, color='purple',
                    ha='right', va='center')
        else:
            ax.annotate(f'steady state = peak {m["vss"]:.1f} mV\n(no sag)',
                        (DUR - 0.5, m['vss']),
                        xytext=(DUR - 90, m['vss'] + 8), fontsize=8,
                        color='purple')

        # offset response
        if m['n_ap']:
            y = 30
            ax.annotate('', (DUR, y), (DUR + m['latency'], y),
                        arrowprops=dict(arrowstyle='<->', color='tab:blue',
                                        lw=0.9, shrinkA=0, shrinkB=0))
            ax.text(DUR - 2, y, f'latency\n{m["latency"]:.1f} ms',
                    fontsize=8, color='tab:blue', va='center', ha='right')
            ax.plot([DUR, DUR + T_POST], [m['ap_thr']] * 2, ls='--',
                    color='tab:blue', lw=0.6)
            ax.text(DUR - 2, m['ap_thr'], f'AP thr. {m["ap_thr"]:.0f} mV',
                    fontsize=8, color='tab:blue', ha='right', va='center')
            ax.text(DUR + T_POST - 2, ybot + 10, f'{m["n_ap"]} offset AP'
                    + ('s' if m['n_ap'] > 1 else ''), fontsize=9,
                    color='tab:blue', ha='right', fontweight='bold')
        else:
            ax.text(DUR + T_POST - 2, ybot + 10, f'no offset AP\nrebound '
                    f'peak {m["rebound"]:.1f} mV', fontsize=9,
                    color='tab:blue', ha='right', fontweight='bold')

    # overlay of the offset responses
    ax = fig.add_subplot(gs[2, :])
    for name, c in res.items():
        ax.plot(c['t'] - DUR, c['v'], color=c['color'], lw=1.2, label=name)
        if annotate and c['m']['n_ap']:
            ax.plot(c['m']['latency'], 46, 'v', color=c['color'], ms=7)
    ax.axvline(0, color='0.5', lw=0.8, ls='--')
    ax.set_xlim(-10, 60)
    ax.set_ylim(ybot, 50)
    ax.set_xlabel('Time from step offset (ms)')
    ax.set_ylabel('Vm (mV)')
    ax.spines[['top', 'right']].set_visible(False)
    if annotate:
        ax.text(0.5, ybot + 5, 'step offset', fontsize=8, color='0.4')
        ax.set_title('(e) Offset responses aligned to step offset '
                     '(triangles = first AP)', loc='left', fontsize=11)
        ax.legend(frameon=False, loc='lower right', ncol=2, fontsize=9)
        fig.suptitle(suptitle, fontsize=12)
    else:
        ax.set_title('(e)', loc='left', fontsize=11)

    fig.savefig(fname, dpi=300 if not annotate else 150, bbox_inches='tight')
    print(f'saved {fname}')


# Figure 1: I_H-blocked steps reduced so Vm settles at V_NOIH (cf. Fig. 7Fii)
res = simulate(match_v=True)
title = (f'single {DUR:.0f} ms step (-400 pA; I_H-blocked: step reduced so '
         f'Vm settles at {V_NOIH:.0f} mV)')
print_table(res, title)
plot(res, 'single_stimulus_conditions.png', annotate=False)
plot(res, 'single_stimulus_conditions_annotated.png', f'SPN model: {title}')

# Figure 2: the same -400 pA step for every condition
res = simulate(match_v=False)
title = f'single {DUR:.0f} ms step, -400 pA for all conditions'
print_table(res, title)
plot(res, 'single_stimulus_conditions_400pA.png', annotate=False)
plot(res, 'single_stimulus_conditions_400pA_annotated.png',
     f'SPN model: {title}')
