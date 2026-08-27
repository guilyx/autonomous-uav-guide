# Connectivity Annulus (composed CBFs)

`SafeDistanceBarrier` pushes every pair apart; `ConnectivityBarrier` pulls
every pair together. Run both and the fleet is squeezed into a shell —
close enough to talk, far enough not to touch — without anything having
planned that shape.

## Result

| quantity | value | limit |
|---|---|---|
| closest pair | **6.00 m** | safe distance 6.0 m |
| furthest pair | **26.00 m** | comm range 26.0 m |

Both barriers active simultaneously, each held exactly at its bound.

Ask for more separation than the radio can span and the QP has no solution:
the impossible case reports infeasible on **2200/2200** steps rather than
quietly returning something unsafe.

The fleet starts *inside* the safe set. A CBF guarantees forward invariance
— it holds a set you are already in — and cannot undo a violated initial
condition.

## Usage

```bash
python -m flybots.simulations.safety.connectivity_annulus
```

## Result

![connectivity_annulus](connectivity_annulus.gif)
