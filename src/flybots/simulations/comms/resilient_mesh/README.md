# Resilient Mesh

A connected network is not a survivable one. At the halfway mark the agent
whose removal costs the most connectivity — the adversarial choice, not a
random one — is switched off.

## Result

| fleet | worst k before → after | λ₂ after |
|---|---|---|
| tight floor (0.55) | **3 → 2** | 0.688 |
| loose floor (0.24) | **1 → 1** | 0.323 |

Neither fleet splits. The tight fleet keeps k ≥ 2 throughout, so it still
survives *any* single further loss; the loose fleet was a single point of
failure before the loss and remains one. Redundancy is the thing being
measured, and it is not the same question as "is the network up".

k is reported as the **worst value over a trailing six seconds**, matching
the plot. It is an integer that flips as agents drift across the link
threshold, so an instantaneous sample flatters whichever moment it lands on.

**k and λ₂ answer different questions.** k is computed on the thresholded
graph; λ₂ on the weighted one. A Gaussian link never reaches exactly zero,
so λ₂ stays positive even for a fleet with no usable links — it merely
becomes very small. The threshold matters: at 0.25 a link here needs 50 m
while the median pair sits 70 m apart, which reads as a disconnected mesh
that is doing fine.

## Usage

```bash
python -m flybots.simulations.comms.resilient_mesh
```

## Result

![resilient_mesh](resilient_mesh.gif)
