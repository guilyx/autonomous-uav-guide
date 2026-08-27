# Convoy Escort

A ground vehicle drives a route away from a base station carrying no
long-range radio. It stays reachable only while a chain of UAVs bridges the
gap — and the gap grows, so the chain must lengthen, then shorten as the
route curves back.

## Result

| escort behaviour | convoy reachable |
|---|---|
| relay chain | **82.1 %** of the run |
| fixed formation around the convoy | 9.2 % |

Max 7 hops at a maximum convoy range of 327 m. Nothing plans the chain:
each aircraft is pulled toward the midpoint of the link it owns and pushed
by the shared connectivity gradient, so the number of hops is an outcome,
not an input.

The fixed formation is the obvious thing to do and loses the base entirely
once the convoy is far enough out.

## Usage

```bash
python -m flybots.simulations.comms.convoy_escort
```

## Result

![convoy_escort](convoy_escort.gif)
