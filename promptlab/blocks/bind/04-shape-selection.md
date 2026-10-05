SHAPE SELECTION:
- Prefer the shape whose five-tuple and constraint slots most precisely match what the
  variable names. Choose a more specific shape (e.g. a gender-specific member) only when the
  variable names that specificity; otherwise prefer the broader/aggregate shape.
- retrieval_score is weak and approximate: a lower-scored shape that matches semantically
  beats a higher-scored shape measuring something else. Never treat it as authoritative, and
  never use score gaps to decide also_plausible.
