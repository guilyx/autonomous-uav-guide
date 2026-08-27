# Link Budget

Same fleet, same controller, same gains. Only the path-loss exponent
changes: n = 2 for free space, n = 4 through clutter.

## Result — a negative one

This was built to show that a fleet tuned on free space fragments when
flown somewhere cluttered. **It does not.**

| configuration | λ₂ |
|---|---|
| matched clutter (n=4 tuned, n=4 flown) | 1.562 |
| mismatched (n=2 tuned, n=4 flown) | **1.557** |

A difference of a third of a percent. The connectivity gradient is
dominated by geometry, not the exponent: both models are smooth and
monotone in range, so they rank links the same way and push nearly the same
direction. Getting the link model wrong costs surprisingly little.

What the exponent *does* change is the margin the same geometry buys:

| | spread | λ₂ |
|---|---|---|
| free space | 94.7 m | **3.98** |
| clutter | 94.6 m | **1.71** |

Identical formation, less than half the connectivity. A margin computed in
one world does not transfer to the other, even though the control does.

## Usage

```bash
python -m flybots.simulations.comms.link_budget
```

## Result

![link_budget](link_budget.gif)
