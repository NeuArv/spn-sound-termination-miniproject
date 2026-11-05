"""Single SPN Vm trace (full model, one -400 pA / 100 ms step) coloured by
phase: rest (purple), inhibition-driven hyperpolarization (red), I_H-driven
sag, rebound and first offset AP (green), I_TCa-supported burst (blue)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from neuron import h

os.chdir(os.path.dirname(os.path.abspath(__file__)))
h.load_file('stdrun.hoc')
h.load_file('spn_neuron.hoc')
seg = h.soma(0.5)

DUR, AMP, T_ON, T_POST = 100.0, -0.4, 200.0, 150.0
T_OFF = T_ON + DUR
stim = h.IClamp(seg)
stim.delay, stim.dur, stim.amp = T_ON, DUR, AMP
t_vec = h.Vector().record(h._ref_t)
v_vec = h.Vector().record(seg._ref_v)
h.dt = 0.01
h.finitialize(-60)
h.continuerun(T_OFF + T_POST)
t = np.array(t_vec) - T_ON
v = np.array(v_vec)

# phase boundaries
i0 = np.searchsorted(t, 0)
during = (t > 0) & (t < DUR)
i_min = np.argmin(np.where(during, v, np.inf))
i_off = np.searchsorted(t, DUR)
i_ap1 = i_off + np.argmax(v[i_off:] > 0)                 # first AP upstroke
# end of the 1st AP = its afterhyperpolarization minimum
i_ahp1 = i_ap1 + np.argmax((v[i_ap1:-1] < -40) & (np.diff(v[i_ap1:]) > 0))

phases = [
    (0, i0, '#9b2d91', 'Rest'),
    (i0, i_min, 'red', 'Inhibition (MNTB)\nhyperpolarizes'),
    (i_min, i_ahp1, 'green', 'I$_H$: sag, rebound\n+ first offset AP'),
    (i_ahp1, len(t), '#0b4f9c', 'I$_{TCa}$: sustained\noffset burst'),
]

fig, ax = plt.subplots(figsize=(13, 5.5))
for a, b, col, name in phases:
    ax.plot(t[a:b + 1], v[a:b + 1], color=col, lw=2.2, label=name)
for x in (0, DUR):
    ax.axvline(x, color='0.5', ls='--', lw=0.8)
ax.text(DUR / 2, 44, 'sound ON', ha='center', fontsize=11, color='0.35',
        fontweight='bold')
ax.text(DUR + 75, 44, 'silence', ha='center', fontsize=11, color='0.35',
        fontweight='bold')
ax.set_xlim(-50, DUR + T_POST)
ax.set_ylim(-110, 52)
ax.set_xlabel('Time from sound onset (ms)', fontsize=12)
ax.set_ylabel('V$_m$ (mV)', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fontsize=10, ncol=4)

fig.savefig('single_trace_phases.png', dpi=300, bbox_inches='tight',
            transparent=True)
print('saved single_trace_phases.png')
