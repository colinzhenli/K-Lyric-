"""Generate K-Lyric architecture diagram as a PNG image."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(18, 9), dpi=200)
ax.set_xlim(0, 18)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#0e1117')

# ── Colors ──
BG = '#0e1117'
SURFACE = '#1a1f2e'
GOLD = '#e8c46c'
BLUE = '#5b9bd5'
GREEN = '#6ec985'
PURPLE = '#b07cd8'
RED = '#e85d5d'
WHITE = '#f0eeea'
DIM = '#7a7d85'
ENC_BG = '#192a4a'
DEC_BG = '#1a3a2a'

def rounded_box(x, y, w, h, color, fill, lw=1.5, alpha=1.0, style='round,pad=0.1'):
    box = FancyBboxPatch((x, y), w, h, boxstyle=style,
                         facecolor=fill, edgecolor=color, linewidth=lw, alpha=alpha,
                         zorder=2, mutation_scale=0.3)
    ax.add_patch(box)

def arrow(x1, y1, x2, y2, color=GOLD, lw=2, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=3)

def text(x, y, s, color=WHITE, size=11, weight='normal', ha='center', va='center', family='sans-serif', style='normal'):
    ax.text(x, y, s, color=color, fontsize=size, fontweight=weight,
            ha=ha, va=va, fontfamily=family, fontstyle=style, zorder=5)

# ═══════════════════════════════════════════════
# ROW 1: MAIN PIPELINE (y ~ 5.5-7.5)
# ═══════════════════════════════════════════════

# ── 1. User Input ──
rounded_box(0.3, 5.5, 2.4, 2.0, GOLD, SURFACE)
text(1.5, 7.1, 'USER INPUT', color=GOLD, size=8, weight='bold', family='monospace')
text(1.5, 6.5, '阳光 , 哀伤', color=WHITE, size=16, weight='bold')
text(1.5, 6.0, '(sunshine, sorrow)', color=DIM, size=9)

# Arrow 1→2
arrow(2.7, 6.5, 3.5, 6.5)

# ── 2. Format + Tokenize ──
rounded_box(3.5, 5.2, 3.2, 2.6, PURPLE, SURFACE)
text(5.1, 7.4, 'FORMAT + TOKENIZE', color=PURPLE, size=8, weight='bold', family='monospace')
text(5.1, 6.8, 'trigger:', color=DIM, size=10, family='monospace')
# Colored token display
ax.text(3.8, 6.2, '<a>', color=BLUE, fontsize=11, fontweight='bold', fontfamily='monospace', zorder=5)
ax.text(4.25, 6.2, '阳光', color=GOLD, fontsize=13, fontweight='bold', zorder=5)
ax.text(4.85, 6.2, '<b>', color=PURPLE, fontsize=11, fontweight='bold', fontfamily='monospace', zorder=5)
ax.text(5.3, 6.2, '哀伤', color=GOLD, fontsize=13, fontweight='bold', zorder=5)
ax.text(5.9, 6.2, '<c>', color=GREEN, fontsize=11, fontweight='bold', fontfamily='monospace', zorder=5)
text(5.1, 5.6, 'Custom tokens added\nto vocabulary', color=DIM, size=8)

# Arrow 2→3
arrow(6.7, 6.5, 7.5, 6.5)

# ── 3. mT5 Model (big dashed box) ──
dashed_box = FancyBboxPatch((7.4, 3.6), 5.6, 4.6, boxstyle='round,pad=0.15',
                             facecolor='none', edgecolor=GOLD, linewidth=1.2,
                             linestyle='--', alpha=0.35, zorder=1)
ax.add_patch(dashed_box)
text(10.2, 8.0, 'mT5-small  (300M params)', color=GOLD, size=11, weight='bold', family='monospace')

# ── 3a. Encoder ──
rounded_box(7.8, 5.9, 4.8, 1.8, BLUE, ENC_BG)
text(10.2, 7.3, 'ENCODER', color=BLUE, size=10, weight='bold', family='monospace')
text(10.2, 6.7, 'Multi-Head Self-Attention', color=WHITE, size=12, weight='bold')
text(10.2, 6.25, 'Feed-Forward Layers × 8', color=DIM, size=10)

# Arrow Encoder → Decoder
arrow(10.2, 5.9, 10.2, 5.3, color=GOLD, lw=1.5)
text(10.9, 5.6, 'hidden\nstates', color=DIM, size=8)

# ── 3b. Decoder ──
rounded_box(7.8, 3.8, 4.8, 1.5, GREEN, DEC_BG)
text(10.2, 5.0, 'DECODER', color=GREEN, size=10, weight='bold', family='monospace')
text(10.2, 4.55, 'Cross-Attention + Autoregressive Generation', color=WHITE, size=10.5, weight='bold')
# Decoder output tokens
ax.text(8.4, 4.05, '<a>', color=BLUE, fontsize=10, fontweight='bold', fontfamily='monospace', zorder=5)
ax.text(8.8, 4.05, '洒在', color=GREEN, fontsize=11, fontweight='bold', zorder=5)
ax.text(9.45, 4.05, '<b>', color=PURPLE, fontsize=10, fontweight='bold', fontfamily='monospace', zorder=5)
ax.text(9.85, 4.05, '的脸上', color=GREEN, fontsize=11, fontweight='bold', zorder=5)
ax.text(10.7, 4.05, '<c>', color=GREEN, fontsize=10, fontweight='bold', fontfamily='monospace', zorder=5)

# Arrow Model → Post-process
arrow(12.6, 4.5, 13.4, 4.5, color=GREEN)

# ── 4. Post-process / Recombine ──
rounded_box(13.4, 3.8, 2.8, 2.8, GREEN, SURFACE)
text(14.8, 6.2, 'RECOMBINE', color=GREEN, size=8, weight='bold', family='monospace')
text(14.8, 5.6, 'Interleave generated\nsegments with\noriginal keywords', color=DIM, size=9)
# Show recombination
text(14.8, 4.7, 'segments + keywords', color=DIM, size=8)
ax.text(13.7, 4.2, '阳光', color=GOLD, fontsize=12, fontweight='bold', zorder=5)
ax.text(14.35, 4.2, '洒在', color=GREEN, fontsize=12, fontweight='bold', zorder=5)
ax.text(14.95, 4.2, '哀伤', color=GOLD, fontsize=12, fontweight='bold', zorder=5)
ax.text(15.55, 4.2, '的脸上', color=GREEN, fontsize=12, fontweight='bold', zorder=5)

# Arrow → Final Output
arrow(14.8, 3.8, 14.8, 3.1, color=GREEN)

# ── 5. Final Output ──
rounded_box(13.0, 1.8, 3.6, 1.3, GREEN, '#1a3028', lw=2.5)
text(14.8, 2.8, 'OUTPUT', color=GREEN, size=8, weight='bold', family='monospace')
text(14.8, 2.3, '阳光洒在哀伤的脸上', color=WHITE, size=16, weight='bold')

# ═══════════════════════════════════════════════
# ROW 2: BOTTOM INFO BOXES (y ~ 0.3-2.5)
# ═══════════════════════════════════════════════

# ── Training Details ──
rounded_box(0.3, 0.3, 5.5, 2.8, GOLD + '40', SURFACE + 'cc')
text(3.05, 2.7, 'TRAINING', color=GOLD, size=9, weight='bold', family='monospace')
text(0.7, 2.2, 'Loss:', color=WHITE, size=10, weight='bold', ha='left')
text(2.0, 2.2, 'Cross-entropy on decoder output', color=DIM, size=10, ha='left')
text(0.7, 1.7, 'Optimizer:', color=WHITE, size=10, weight='bold', ha='left')
text(2.5, 1.7, 'AdamW (lr=1e-5, wd=0.01)', color=DIM, size=10, ha='left')
text(0.7, 1.2, 'Hardware:', color=WHITE, size=10, weight='bold', ha='left')
text(2.5, 1.2, '4× Tesla V100, ~30 epochs (~1.5 days)', color=DIM, size=10, ha='left')
text(0.7, 0.7, 'Decoding:', color=WHITE, size=10, weight='bold', ha='left')
text(2.5, 0.7, 'Beam search (10 beams, rep. penalty=10)', color=DIM, size=10, ha='left')

# ── Key Insight ──
rounded_box(6.3, 0.3, 6.2, 2.8, GOLD + '50', GOLD + '0a')
text(9.4, 2.7, 'KEY INSIGHT', color=GOLD, size=9, weight='bold', family='monospace')
text(9.4, 2.15, 'Our task mirrors T5\'s span corruption pre-training:', color=WHITE, size=11, weight='bold')

text(6.7, 1.5, 'Pre-train:', color=DIM, size=10, ha='left', weight='bold')
ax.text(8.2, 1.5, 'The <X> walks in <Y> park → ', color=DIM, fontsize=10, ha='left',
        fontfamily='monospace', zorder=5, va='center')
ax.text(11.2, 1.5, 'cat, the', color=BLUE, fontsize=10, ha='left',
        fontfamily='monospace', fontweight='bold', zorder=5, va='center')

text(6.7, 0.95, 'Ours:', color=GOLD, size=10, ha='left', weight='bold')
ax.text(8.2, 0.95, '<a>阳光<b>哀伤<c>  →  ', color=DIM, fontsize=10, ha='left',
        fontfamily='monospace', zorder=5, va='center')
ax.text(10.85, 0.95, '洒在, 的脸上', color=GREEN, fontsize=10, ha='left',
        fontfamily='monospace', fontweight='bold', zorder=5, va='center')

text(9.4, 0.5, 'Fine-tuning objective ≈ pre-training objective → efficient transfer', color=GOLD, size=10, style='italic')

# ── Title ──
text(9.0, 8.7, 'K-Lyric: Method Overview', color=WHITE, size=18, weight='bold', family='serif')

plt.tight_layout(pad=0.3)
plt.savefig('/local-scratch2/ericw/K-Lyric-/method_diagram.png',
            facecolor=BG, edgecolor='none', bbox_inches='tight', dpi=200)
print("Saved method_diagram.png")

# Also generate a white-background version for slides
fig2, ax2 = plt.subplots(1, 1, figsize=(18, 9), dpi=200)
ax2.set_xlim(0, 18)
ax2.set_ylim(0, 9)
ax2.set_aspect('equal')
ax2.axis('off')
fig2.patch.set_facecolor('#ffffff')

W_SURFACE = '#f5f5f8'
W_GOLD = '#c4960a'
W_BLUE = '#2563a8'
W_GREEN = '#1a8a40'
W_PURPLE = '#7c3aad'
W_WHITE = '#1a1a1a'
W_DIM = '#666666'
W_ENC_BG = '#e8f0fa'
W_DEC_BG = '#e6f5ec'

def rb2(x, y, w, h, color, fill, lw=1.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                         facecolor=fill, edgecolor=color, linewidth=lw, zorder=2)
    ax2.add_patch(box)

def ar2(x1, y1, x2, y2, color=W_GOLD, lw=2):
    ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=3)

def tx2(x, y, s, color=W_WHITE, size=11, weight='normal', ha='center', va='center', family='sans-serif', style='normal'):
    ax2.text(x, y, s, color=color, fontsize=size, fontweight=weight,
            ha=ha, va=va, fontfamily=family, fontstyle=style, zorder=5)

# ── 1. User Input ──
rb2(0.3, 5.5, 2.4, 2.0, W_GOLD, W_SURFACE)
tx2(1.5, 7.1, 'USER INPUT', color=W_GOLD, size=8, weight='bold', family='monospace')
tx2(1.5, 6.5, '阳光 , 哀伤', color=W_WHITE, size=16, weight='bold')
tx2(1.5, 6.0, '(sunshine, sorrow)', color=W_DIM, size=9)

ar2(2.7, 6.5, 3.5, 6.5)

# ── 2. Format ──
rb2(3.5, 5.2, 3.2, 2.6, W_PURPLE, W_SURFACE)
tx2(5.1, 7.4, 'FORMAT + TOKENIZE', color=W_PURPLE, size=8, weight='bold', family='monospace')
tx2(5.1, 6.8, 'trigger:', color=W_DIM, size=10, family='monospace')
ax2.text(3.8, 6.2, '<a>', color=W_BLUE, fontsize=11, fontweight='bold', fontfamily='monospace', zorder=5)
ax2.text(4.25, 6.2, '阳光', color=W_GOLD, fontsize=13, fontweight='bold', zorder=5)
ax2.text(4.85, 6.2, '<b>', color=W_PURPLE, fontsize=11, fontweight='bold', fontfamily='monospace', zorder=5)
ax2.text(5.3, 6.2, '哀伤', color=W_GOLD, fontsize=13, fontweight='bold', zorder=5)
ax2.text(5.9, 6.2, '<c>', color=W_GREEN, fontsize=11, fontweight='bold', fontfamily='monospace', zorder=5)
tx2(5.1, 5.6, 'Custom tokens added\nto vocabulary', color=W_DIM, size=8)

ar2(6.7, 6.5, 7.5, 6.5)

# ── 3. mT5 ──
dashed2 = FancyBboxPatch((7.4, 3.6), 5.6, 4.6, boxstyle='round,pad=0.15',
                           facecolor='none', edgecolor=W_GOLD, linewidth=1.2,
                           linestyle='--', alpha=0.4, zorder=1)
ax2.add_patch(dashed2)
tx2(10.2, 8.0, 'mT5-small  (300M params)', color=W_GOLD, size=11, weight='bold', family='monospace')

rb2(7.8, 5.9, 4.8, 1.8, W_BLUE, W_ENC_BG)
tx2(10.2, 7.3, 'ENCODER', color=W_BLUE, size=10, weight='bold', family='monospace')
tx2(10.2, 6.7, 'Multi-Head Self-Attention', color=W_WHITE, size=12, weight='bold')
tx2(10.2, 6.25, 'Feed-Forward Layers × 8', color=W_DIM, size=10)

ar2(10.2, 5.9, 10.2, 5.3, color=W_GOLD, lw=1.5)
tx2(10.9, 5.6, 'hidden\nstates', color=W_DIM, size=8)

rb2(7.8, 3.8, 4.8, 1.5, W_GREEN, W_DEC_BG)
tx2(10.2, 5.0, 'DECODER', color=W_GREEN, size=10, weight='bold', family='monospace')
tx2(10.2, 4.55, 'Cross-Attention + Autoregressive Generation', color=W_WHITE, size=10.5, weight='bold')
ax2.text(8.4, 4.05, '<a>', color=W_BLUE, fontsize=10, fontweight='bold', fontfamily='monospace', zorder=5)
ax2.text(8.8, 4.05, '洒在', color=W_GREEN, fontsize=11, fontweight='bold', zorder=5)
ax2.text(9.45, 4.05, '<b>', color=W_PURPLE, fontsize=10, fontweight='bold', fontfamily='monospace', zorder=5)
ax2.text(9.85, 4.05, '的脸上', color=W_GREEN, fontsize=11, fontweight='bold', zorder=5)
ax2.text(10.7, 4.05, '<c>', color=W_GREEN, fontsize=10, fontweight='bold', fontfamily='monospace', zorder=5)

ar2(12.6, 4.5, 13.4, 4.5, color=W_GREEN)

# ── 4. Recombine ──
rb2(13.4, 3.8, 2.8, 2.8, W_GREEN, W_SURFACE)
tx2(14.8, 6.2, 'RECOMBINE', color=W_GREEN, size=8, weight='bold', family='monospace')
tx2(14.8, 5.6, 'Interleave generated\nsegments with\noriginal keywords', color=W_DIM, size=9)
tx2(14.8, 4.7, 'segments + keywords', color=W_DIM, size=8)
ax2.text(13.7, 4.2, '阳光', color=W_GOLD, fontsize=12, fontweight='bold', zorder=5)
ax2.text(14.35, 4.2, '洒在', color=W_GREEN, fontsize=12, fontweight='bold', zorder=5)
ax2.text(14.95, 4.2, '哀伤', color=W_GOLD, fontsize=12, fontweight='bold', zorder=5)
ax2.text(15.55, 4.2, '的脸上', color=W_GREEN, fontsize=12, fontweight='bold', zorder=5)

ar2(14.8, 3.8, 14.8, 3.1, color=W_GREEN)

# ── 5. Output ──
rb2(13.0, 1.8, 3.6, 1.3, W_GREEN, W_DEC_BG, lw=2.5)
tx2(14.8, 2.8, 'OUTPUT', color=W_GREEN, size=8, weight='bold', family='monospace')
tx2(14.8, 2.3, '阳光洒在哀伤的脸上', color=W_WHITE, size=16, weight='bold')

# ── Training ──
rb2(0.3, 0.3, 5.5, 2.8, '#cccccc', '#fafafa')
tx2(3.05, 2.7, 'TRAINING', color=W_GOLD, size=9, weight='bold', family='monospace')
tx2(0.7, 2.2, 'Loss:', color=W_WHITE, size=10, weight='bold', ha='left')
tx2(2.0, 2.2, 'Cross-entropy on decoder output', color=W_DIM, size=10, ha='left')
tx2(0.7, 1.7, 'Optimizer:', color=W_WHITE, size=10, weight='bold', ha='left')
tx2(2.5, 1.7, 'AdamW (lr=1e-5, wd=0.01)', color=W_DIM, size=10, ha='left')
tx2(0.7, 1.2, 'Hardware:', color=W_WHITE, size=10, weight='bold', ha='left')
tx2(2.5, 1.2, '4× Tesla V100, ~30 epochs (~1.5 days)', color=W_DIM, size=10, ha='left')
tx2(0.7, 0.7, 'Decoding:', color=W_WHITE, size=10, weight='bold', ha='left')
tx2(2.5, 0.7, 'Beam search (10 beams, rep. penalty=10)', color=W_DIM, size=10, ha='left')

# ── Key Insight ──
rb2(6.3, 0.3, 6.2, 2.8, W_GOLD + '80', '#fdf8eb')
tx2(9.4, 2.7, 'KEY INSIGHT', color=W_GOLD, size=9, weight='bold', family='monospace')
tx2(9.4, 2.15, 'Our task mirrors T5\'s span corruption pre-training:', color=W_WHITE, size=11, weight='bold')
tx2(6.7, 1.5, 'Pre-train:', color=W_DIM, size=10, ha='left', weight='bold')
ax2.text(8.2, 1.5, 'The <X> walks in <Y> park → ', color=W_DIM, fontsize=10, ha='left', fontfamily='monospace', zorder=5, va='center')
ax2.text(11.2, 1.5, 'cat, the', color=W_BLUE, fontsize=10, ha='left', fontfamily='monospace', fontweight='bold', zorder=5, va='center')
tx2(6.7, 0.95, 'Ours:', color=W_GOLD, size=10, ha='left', weight='bold')
ax2.text(8.2, 0.95, '<a>阳光<b>哀伤<c>  →  ', color=W_DIM, fontsize=10, ha='left', fontfamily='monospace', zorder=5, va='center')
ax2.text(10.85, 0.95, '洒在, 的脸上', color=W_GREEN, fontsize=10, ha='left', fontfamily='monospace', fontweight='bold', zorder=5, va='center')
tx2(9.4, 0.5, 'Fine-tuning objective ≈ pre-training objective → efficient transfer', color=W_GOLD, size=10, style='italic')

# Title
tx2(9.0, 8.7, 'K-Lyric: Method Overview', color=W_WHITE, size=18, weight='bold', family='serif')

plt.tight_layout(pad=0.3)
plt.savefig('/local-scratch2/ericw/K-Lyric-/method_diagram_white.png',
            facecolor='#ffffff', edgecolor='none', bbox_inches='tight', dpi=200)
print("Saved method_diagram_white.png")
