<!-- Erwin Lejeune — 2026-08-27 -->
# Safety

Safety here means a set the vehicle is not allowed to leave, enforced at the
last moment before the command reaches the actuators. The chapter is built
around control barrier functions: a barrier defines the safe set, and a
small quadratic program edits whatever the nominal controller asked for by
the least amount that keeps the set invariant.

## Why filter rather than plan

A planner can be made to respect obstacles, but only against the world it
was given, and only until something moves. A filter makes no plan at all. It
sits between an existing controller and the vehicle and refuses the parts of
the command that would leave the safe set, which means safety can be argued
about separately from performance — and the controller underneath can stay
as simple, or as learned, as you like.

The cost is that a filter is *reactive*. It has no way to route around a
problem, only to decline; and because standing still satisfies every barrier
here, a filter can be perfectly safe and completely stuck. Both properties
are visible in the position-exchange simulation.

## Core Questions

- What is the relative degree of the constraint, and does the input actually
  appear in the derivative being conditioned?
- Who is responsible for avoiding whom, and does the formulation split that
  effort or duplicate it?
- What should happen when the constraints admit no input at all?

## Algorithms

- [Position Exchange](/simulations/safety/position-exchange)

## Prerequisites

- Quadratic programming and constrained least squares
- Lie derivatives and relative degree
- Set invariance and class-K functions
