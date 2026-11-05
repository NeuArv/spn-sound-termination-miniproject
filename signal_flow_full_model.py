"""Signal flow through the sound-offset circuit for one stimulus (full model).

Colour code follows the circuit slide (slide 3):
  sound = grey, primary receiver = black, inhibitor unit (MNTB) = red,
  inhibitory synapse = blue, offset processor (SPN) = purple.
I_H (teal) and I_TCa (orange) use the colours of the slide 6 annotations.

The SPN trace and its I_H / I_TCa currents are simulated with the full model
and the same -400 pA, 100 ms step as slide 6 (a); the step stands in for the
summed glycinergic IPSC. The primary-receiver and MNTB spike trains are
schematic (not simulated).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from neuron import h

os.chdir(os.path.dirname(os.path.abspath(__file__)))
h.load_file('stdrun.hoc')
h.load_file('spn_neuron.hoc')
seg = h.soma(0.5)

DUR, AMP, T_ON, T_POST = 100.0, -0.4, 200.0, 150.0
T_OFF = T_ON + DUR
C = dict(sound='0.45', recv='k', mntb='red', syn='blue', spn='#9b2d91',
         ih='#00897b', ica='#e69f00')

stim = h.IClamp(seg)
stim.delay, stim.dur, stim.amp = T_ON, DUR, AMP
rec = {k: h.Vector().record(r) for k, r in dict(
    t=h._ref_t, v=seg._ref_v, ih=seg._ref_i_sjg_ih, ica=seg._ref_ica).items()}
spikes = h.Vector()
nc = h.NetCon(seg._ref_v, None, sec=h.soma)
nc.threshold = 0
nc.record(spikes)
h.dt = 0.01
h.finitialize(-60)
h.continuerun(T_OFF + T_POST)

area_cm2 = seg.area() * 1e-8
r = {k: np.array(x) for k, x in rec.items()}
t = r['t'] - T_ON
v = r['v']
ih = r['ih'] * area_cm2 * 1e9          # mA/cm2 * cm2 = mA -> pA
ica = r['ica'] * area_cm2 * 1e9
sp = np.array(spikes) - T_ON
off_sp = sp[sp > DUR]
lat = off_sp[0] - DUR
v_rest = v[np.searchsorted(t, -1)]

# schematic spike trains (phase-locked firing during the sound)
rng = np.random.default_rng(3)
recv_sp = np.arange(2.0, DUR - 1.0, 3.0) + rng.normal(0, 0.3, 33)
mntb_sp = recv_sp + 1.0 + rng.normal(0, 0.15, recv_sp.size)

fig, axs = plt.subplots(6, 1, figsize=(13, 9.5), sharex=True,
                        gridspec_kw=dict(height_ratios=[0.55, 0.45, 0.45,
                                                        0.8, 2.2, 1.4],
                                         hspace=0.25))
XL = (-50, 250)
for ax in axs:
    ax.axvspan(0, DUR, color='0.93', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(*XL)


def label(ax, text, color):
    ax.text(-0.075, 0.5, text, transform=ax.transAxes, ha='right',
            va='center', color=color, fontsize=11, fontweight='bold')


# 1 sound
ax = axs[0]
ts = np.linspace(0, DUR, 4000)
ax.plot(ts, np.sin(2 * np.pi * 0.6 * ts), color=C['sound'], lw=0.5)
ax.plot([XL[0], 0], [0, 0], [DUR, XL[1]], [0, 0], color=C['sound'], lw=1)
ax.set_ylim(-1.3, 1.3)
ax.axis('off')
label(ax, 'Source stimulus\n(sound)', C['sound'])
ax.text(DUR / 2, 1.45, 'SOUND ON', ha='center', fontsize=10,
        fontweight='bold', color='0.3')
ax.text(DUR + 75, 1.45, 'SILENCE', ha='center', fontsize=10,
        fontweight='bold', color='0.3')

# 2-3 spike trains
for ax, s, col, name in [(axs[1], recv_sp, C['recv'], 'Primary receiver'),
                         (axs[2], mntb_sp, C['mntb'], 'Inhibitor unit\n(MNTB)')]:
    ax.vlines(s, 0, 1, color=col, lw=1.3)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    label(ax, name, col)
axs[2].text(DUR + 3, 0.5, 'stops at sound offset', color=C['mntb'],
            fontsize=9, va='center')

# 4 inhibitory synaptic current
ax = axs[3]
i_syn = np.where((t >= 0) & (t < DUR), AMP * 1e3, 0.0)
ax.plot(t, i_syn, color=C['syn'], lw=2)
ax.fill_between(t, i_syn, 0, color=C['syn'], alpha=0.25, lw=0)
ax.set_ylim(-480, 80)
ax.set_yticks([-400, 0])
ax.set_ylabel('pA', fontsize=9, labelpad=2)
label(ax, 'Inhibitory\nsynapse', C['syn'])
ax.text(DUR / 2, -330, 'glycinergic inhibition (modelled as -400 pA)',
        ha='center', color=C['syn'], fontsize=9, fontweight='bold')

# 5 SPN membrane potential
ax = axs[4]
ax.plot(t, v, color=C['spn'], lw=1.3)
ax.axhline(v_rest, ls=':', color='0.5', lw=0.8)
ax.text(-48, v_rest + 4, f'rest {v_rest:.0f} mV', fontsize=8, color='0.4')
ax.set_ylim(-115, 50)
ax.set_ylabel('Vm (mV)')
label(ax, 'Offset processor\n(SPN)', C['spn'])
i_min = np.argmin(np.where((t > 0) & (t < DUR), v, np.inf))
ax.annotate('1  hyperpolarization\n    towards E$_{Cl}$', (t[i_min], v[i_min]),
            xytext=(12, -112), fontsize=9, color=C['syn'], va='bottom')
ax.annotate('2  sag: I$_H$ depolarizes', (60, v[np.searchsorted(t, 60)]),
            xytext=(30, -40), fontsize=9, color=C['ih'],
            arrowprops=dict(arrowstyle='->', color=C['ih']))
ax.annotate('', (DUR, 30), (DUR + lat, 30),
            arrowprops=dict(arrowstyle='<->', color='crimson', lw=1,
                            shrinkA=0, shrinkB=0))
ax.text(DUR - 2, 30, f'latency {lat:.1f} ms', ha='right', va='center',
        fontsize=9, color='crimson', fontweight='bold')
ax.text(off_sp[-1] + 6, 15, f'3  offset burst\n    ({off_sp.size} APs)',
        fontsize=9, color=C['spn'], fontweight='bold')

# 6 intrinsic currents
ax = axs[5]
ax.axhline(0, color='0.6', lw=0.6)
ax.plot(t, ih, color=C['ih'], lw=1.6, label='I$_H$ (HCN)')
ax.plot(t, ica, color=C['ica'], lw=1.6, label='I$_{TCa}$ (T-type Ca$^{2+}$)')
lo = min(ih.min(), ica.min())
ax.set_ylim(lo * 1.15, -lo * 0.15)
ax.set_ylabel('pA (inward < 0)', fontsize=9, labelpad=2)
label(ax, 'Intrinsic\ncurrents', '0.2')
ax.legend(frameon=False, loc='lower right', fontsize=9, ncol=2)
ax.set_xlabel('Time from sound onset (ms)')

for ax in axs:
    ax.axvline(DUR, color='0.5', lw=0.8, ls='--', zorder=1)

fig.savefig('signal_flow_full_model.png', dpi=300, bbox_inches='tight')
print(f'rest {v_rest:.1f} mV, min {v[i_min]:.1f} mV, latency {lat:.1f} ms, '
      f'{off_sp.size} offset APs, I_H peak {ih.min():.0f} pA, '
      f'I_TCa peak {ica.min():.0f} pA')
