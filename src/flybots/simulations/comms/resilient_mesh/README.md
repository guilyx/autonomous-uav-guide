# Resilient Mesh

A connected network is not a survivable one. At the halfway mark the agent
whose removal costs the most connectivity — the adversarial choice, not a
random one — is switched off.

## Result

| fleet | k before → after | λ₂ after |
|---|---|---|
| tight floor (0.55) | **3 → 3** | 0.688 |
| loose floor (0.24) | **2 → 1** | 0.323 |

Neither fleet splits. What the loss costs is *margin*: the loose fleet is
now one further failure from fragmenting, the tight one is no more fragile
than it started.

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
