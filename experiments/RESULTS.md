# Experiment ledger

The trail behind the final ensemble. The headline lever was a custom EfficientNetV2-s
branch trained on multi-teacher pseudo-labels (E14); most other ideas were neutral or
negative on the leaderboard. Keeping the dead-ends here on purpose — knowing what
*didn't* move the needle (and why) was most of the work.

Base pipeline (borrowed, public): Perch v2 + ProtoSSMv5 + distilled SED ≈ **0.949** public.

| Exp | Change | Local OOF | Public LB | Verdict |
|-----|--------|-----------|-----------|---------|
| **E14** | **V2-s branch: broader pseudos (Perch+SED+BirdNET) + labeled soundscapes + balanced sampler, blended 45/30/25** | **+** | **~0.953** | ✅ **the win** |
| — | Final selected submission `cl2tta_e14_integrated` | — | 0.95117 | ✅ **0.95224 private (best)** |
| E20 | Temporal-shift TTA — arithmetic 3-shift | ~0 | 0.953 | ✅ kept |
| E20 | Temporal-shift TTA — rank-space 3-shift | ~0 | 0.952 | ❌ worse than arith |
| — | Expand pseudos to full 10,592 soundscapes (5× data) | +0.0016 | 0.953 | ➖ neutral on LB |
| E30 | Stronger V2-s (more epochs) | +0.0047 | 0.952 | ❌ bigger OOF, worse LB — overfit biased heldout |
| E28 | `min(SED, V2-s)` aggregation | — | 0.951 | ❌ suppresses true positives |
| E18 | nfnet-l0 as a 4th branch | 0.928 | 0.951 | ❌ too weak + correlated |
| E21 | EfficientNetV2-B3 branch | — | — | ❌ correlated with V2-s |
| E24 | Taxonomy (genus/class) probability smoothing | ~0 | 0.953 | ➖ V2-s already captures it |
| E25 | TopK-masked BirdNET sidecar | — | — | ❌ inference > 90 min (timeout) |
| E12 | Sydorskyy-recipe V2-s, higher pseudo emphasis | ~0 | — | ❌ pseudo coverage too low at sp=0.6 |
| v946 | Early pseudo-labeling (full-window, single-round) | −0.025 | — | ❌ deprecated; fixed in E13+ (mid-windows, multi-teacher) |

## Lessons that shaped the rest of the project

- **OOF↔LB transfer was weak here.** A bigger OOF gain (E30) *lost* on the LB — the
  16-file heldout was site/time-biased, so OOF improvements didn't generalize. After
  this I stopped optimizing on raw OOF and treated the public LB as a small,
  high-variance sample to hedge against, not chase.
- **A genuinely new, decorrelated signal beats tuning.** Adding the V2-s branch (a
  different backbone + different training data via pseudos) was the only thing that broke
  past the base 0.949; post-processing tweaks, alternative aggregations, and correlated
  4th models were all neutral-to-negative.
- **The CPU budget is a hard constraint.** The sidecar idea (E25) was promising on paper
  but blew the 90-minute cap — runtime is a first-class design variable, not an
  afterthought.
