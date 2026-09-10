# ATS model builders

Documentation for `src/cheb_ar/models/ats.py`: the driven, dissipative
full-ATS (Asymmetrically Threaded SQUID) system — storage mode `a` (dim
`n_a`) coupled to a lossy buffer mode `b` (dim `n_b`) — in three frames.
The module is the **single source of truth** for the default experimental
parameters; scripts and notebooks must import them, never redefine them.
Importing the module enables `jax_enable_x64` and
`dq.set_precision("double")`.

## 1. Default experimental constants

| Constant | Value | Meaning |
|---|---|---|
| `W_A` | `25.338776456203686` | storage-mode frequency `w_a` (`w_b = 2*w_a`) |
| `KAPPA_B` | `5 / 10.4` | buffer single-photon loss rate |
| `E_J` | `37 * 2*pi` | Josephson energy |
| `PHI_A` | `0.11` | zero-point phase fluctuation, storage mode |
| `PHI_B` | `0.204` | zero-point phase fluctuation, buffer mode |

These encode real experimental parameters — don't change them (or the derived
formulas below) without flagging it.

## 2. Derived quantities (`_derived_params`)

Shared by all three builders; returned as the `params` bookkeeping dict stored
alongside every sweep result:

- `g = sin(epsilon_p) * E_J * phi_a**2 * phi_b` (two-photon coupling)
- `g2 = J_1(epsilon_p) * E_J * phi_a**2 * phi_b` (`scipy.special.jv`)
- `kappa_2 = 4*g**2 / kappa_b` (engineered two-photon dissipation)
- `kappa_1 = 0.005 * kappa_2` (single-photon loss)
- `epsilon_d = 2 * alpha_sq * g2` (drive amplitude fixing the cat size)
- `T_drive = 2*pi / w_a`, `T_block = n_periods * T_drive` (Floquet block)

## 3. The three frame variants

All three return the jump operators `[sqrt(kappa_1) a, sqrt(kappa_b) b]`
(time-dependent in the interaction frame) and `(T_block, params)`. The
non-linear Josephson term is `dq.sinm(phi_a_tot + phi_b_tot) - phi_a_tot -
phi_b_tot`, weighted by `-2 * E_J * sin(epsilon_p)`.

| Builder | Frame | Hamiltonian | Notes |
|---|---|---|---|
| `build_ats_hamiltonian` | lab | `H_0 = w_a a†a + w_b b†b`, drive `epsilon_d cos(w_b t) (b + b†)`, non-linear term | returns `(Ham, jump_ops, T_block, params)` |
| `build_ats_hamiltonian_rotating` | rotating | modulated `a`/`b` at `w_a`/`w_b` inside the non-linear term, static drive `epsilon_d (b + b†) / 2`, no `H_0` | previously `build_ats_hamiltonian_r`; used by `exact_diagonalization.py` |
| `build_ats_hamiltonian_interaction` | interaction frame of the *static* Hamiltonian | drive only (see below) | pairs with `ChebAr`'s `jump_ops_LdL` + `output_phase`; used by both sweep scripts |

Default-argument quirks, preserved as found in the pre-merge code:
`epsilon_p` defaults to `0.3` in the rotating variant but `0.1` in the other
two; `n_a` defaults to `25` in the interaction variant but `20` in the other
two. Sweep scripts pass these explicitly, so the defaults only matter for
interactive use.

### Interaction-frame construction

The static Hamiltonian `H_s = H_0 - 2 E_J sin(epsilon_p) * non_linear_op` is
numerically symmetrized and diagonalized once (`numpy.linalg.eigh`), giving
`H_s = V diag(lam) V†`. Everything downstream lives in that eigenbasis:

- `a_bar = V† a V` (and `b_bar`) are built once; the time dependence is a
  pure phase, `a_tilde(t) = a_bar * exp(1j * Delta * t)` with
  `Delta = lam[:, None] - lam[None, :]`, wrapped in `dq.timecallable`.
- `H_I(t) = epsilon_d * cos(w_b t) * (b_tilde(t) + b_tilde(t)†)` — the drive
  is the only Hamiltonian left.
- `jump_ops_LdL_I = [kappa_1 * (a†a)~(t), kappa_b * (b†b)~(t)]` are the
  precomputed `L†L` operators; passing them as `jump_ops_LdL` to `ChebAr`
  switches the propagator to `dq.mesolve_fast` (pinned dynamiqs fork).
- `output_phase = exp(-1j * lam * T_block)` undoes the frame rotation after
  each block (pass as `output_phase` to `ChebAr`).
- `V` is returned so states can be moved between eigenbases (below).

Returns `(H_I, jump_ops_I, jump_ops_LdL_I, output_phase, V, T_block, params)`.

## 4. `transform_vectorized_state(x_vec, V_from, V_to)`

Re-expresses a column-major (`order="F"`) vectorized operator written in
eigenbasis `V_from` into eigenbasis `V_to`, via the lab frame:
`rho_to = V_to† (V_from rho V_from†) V_to`. Used to warm-start a sweep point
from the previous point's Ritz vector when the static Hamiltonian — hence its
eigenbasis — changes along the sweep (an `eps_p` sweep). For an `alpha_sq`
sweep the basis is unchanged (`alpha_sq` only enters the drive), so the sweep
script reuses vectors as-is.
