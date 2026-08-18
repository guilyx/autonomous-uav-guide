# Security Policy

## Scope

`flybots` is a simulation and educational library. It does not process
untrusted network input, handle credentials, or run privileged operations.
The realistic security surface is small, but two categories matter:

1. **Code execution via loaded artefacts.** Trained policies are loaded
   from `.npz` files. These are read with `allow_pickle=False`, so a
   malicious file cannot execute code — if you find a path where it can,
   that is a genuine vulnerability.
2. **Dependency vulnerabilities** in NumPy, SciPy or Matplotlib that this
   project's usage exposes.

## Supported versions

The latest release on the `main` branch receives fixes. This project has
not yet reached 1.0; older versions are not patched.

| Version | Supported |
|---|---|
| 0.2.x | yes |
| < 0.2 | no |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Report it privately through
[GitHub Security Advisories](https://github.com/guilyx/autonomous-uav-guide/security/advisories/new),
or by email to **erwin.lejeune15@gmail.com**.

Include:

- What the issue is and where in the code it lives
- A reproduction — a script is ideal
- What an attacker could achieve

You can expect an acknowledgement within 7 days and an assessment within
30 days. If the report is valid we will agree a disclosure timeline with
you, and credit you in the advisory unless you prefer otherwise.

## A note on flight safety

This library simulates aircraft. It is a teaching and research tool: the
models are simplified, the controllers are not certified, and nothing here
has been validated for use on a real airframe. **Do not fly hardware on
control code taken from this repository without independent verification
and appropriate safety testing.** Bugs in this repository are software
bugs; bugs in flight code are a different kind of problem, and that
distinction is yours to maintain.
