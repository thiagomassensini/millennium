#!/usr/bin/env python3
"""
Generate Yang-Mills mass gap discretization figure
Shows binary energy spectrum E_k = E_0 * 2^(-k)
"""

import matplotlib.pyplot as plt
import numpy as np

# Energy scales
k = np.arange(0, 11)
E0 = 1000.0  # GeV (TeV scale)
E_k = E0 * 2.0**(-k)

# Create figure
plt.figure(figsize=(10, 6))
plt.semilogy(k, E_k, 'o-', linewidth=2, markersize=10, color='#2E86AB', label='$E_k = E_0 \cdot 2^{-k}$')

# Mark important energy scales
plt.axhline(y=1, color='#A23B72', linestyle='--', linewidth=2, alpha=0.7, label='QCD confinement (~1 GeV)')
plt.axhline(y=125, color='#F18F01', linestyle='--', linewidth=2, alpha=0.7, label='Higgs mass (~125 GeV)')
plt.axhline(y=250, color='#C73E1D', linestyle='--', linewidth=2, alpha=0.7, label='Electroweak scale (~250 GeV)')

# Annotations
plt.text(0.5, 1000, '$E_0$ = 1 TeV', fontsize=10, verticalalignment='bottom')
plt.text(2.5, 250, 'Higgs region', fontsize=9, verticalalignment='bottom')
plt.text(10.5, 1, 'QCD\nconfinement', fontsize=9, verticalalignment='top', horizontalalignment='right')

# Labels and formatting
plt.xlabel('Binary scale $k$', fontsize=13)
plt.ylabel('Energy (GeV)', fontsize=13)
plt.title('Binary Discretized Mass Gap Spectrum\n$E_k = E_0 \cdot 2^{-k}$ with $E_0 = 1$ TeV', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, which='both')
plt.legend(fontsize=10, loc='upper right')
plt.ylim(0.5, 2000)
plt.xlim(-0.5, 10.5)
plt.tight_layout()

# Save
output_path = '/home/thlinux/relacionalidadegeral/papers/figures/yang_mills_mass_gap.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.close()
