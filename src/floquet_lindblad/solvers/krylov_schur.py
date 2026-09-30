"""Thick-restarted Arnoldi (Krylov-Schur) on a vectorized Lindbladian propagator.

``KrylovSchurLindblad`` targets the same quantity as
:class:`~floquet_lindblad.solvers.arnoldi_no_cheb.ArnoldiLindblad` -- the eigenvalue of
largest real part of the trace-projected one-block propagator ``P`` -- with the
same interaction-frame hooks (``jump_ops_LdL``, ``output_phase``) and the same
vectorization conventions. The difference is what happens when the Krylov basis
reaches ``m`` vectors: instead of stopping (``ArnoldiLindblad``) or filtering
with a fixed polynomial (``ChebAr``), the basis is *truncated* to the ``k``
Ritz vectors closest to the unit circle and Arnoldi continues from the residual
direction. No propagator application is wasted at a restart.

Why this helps for one eigenvalue
---------------------------------
Keeping the ``k - 1`` Ritz vectors next to the target across restarts deflates
them: the target then converges as if those eigenvalues were absent from the
spectrum (Morgan 1996). For the cat qubit this matters twice. The phase-flip
pair sits within a factor of a few of the target's distance to 1 at small
``alpha_sq`` and would otherwise stall a single-vector restart; and the
outermost bulk modes, once kept, lower the effective bulk radius that sets the
number of applications per digit. The restart is mathematically the implicit
restart with exact shifts at the discarded Ritz values -- the polynomial in
``P`` one would otherwise build by hand -- applied for free.

Conventions
-----------
* Column-major vectorization, as in ``ArnoldiLindblad``: ``vec(M) = M.T.reshape(-1)``
  and ``unvec(v) = v.reshape(N, N).T``.
* Basis ``Q`` has shape ``(m+1, dim)``; row ``j`` is the ``j``-th basis vector.
  Rows not yet filled are exactly zero, which the two-pass classical
  Gram-Schmidt relies on (their inner products vanish, so no masking).
* ``B`` has shape ``(m+1, m)`` and holds the Krylov-Schur decomposition
  ``P Q[:m].T = Q[:m].T B[:m, :m] + Q[m] B[m, :]``. After a fresh Arnoldi cycle
  ``B[:m, :m]`` is Hessenberg and ``B[m, :]`` has a single entry; after a
  restart the leading ``k x k`` block is dense and row ``k`` carries the
  coupling vector.
* The Krylov vectors are complex combinations after the first restart (the
  kept Ritz vectors of a complex pair are not Hermitian), so unlike
  ``ArnoldiLindblad`` the projected matrix is not real up to noise. The
  imaginary part of the target Ritz value is the noise indicator instead.

Typical use
-----------
>>> solver = KrylovSchurLindblad(H, jump_ops, T_block, dims=(25, 11),
...                              rtol=1e-7, atol=1e-8)          # loop tolerance
>>> x0 = solver.make_x0(seed=0)
>>> out = solver.solve(x0, m=40, k=10, res_tol=1e-6)            # loose cycles
>>> solver.set_tolerance(rtol=1e-9, atol=1e-10)                 # tighten
>>> out = solver.solve(state=out["state"], m=40, k=10, res_tol=1e-8)
>>> final = solver.residual_check(out["x_ritz"], n_blocks=10)   # long block
>>> final["rate_RQ"]
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import dynamiqs as dq

dq.set_precision("double")

COMPLEX = jnp.complex128
REAL = jnp.float64


class KrylovSchurLindblad:
    """Thick-restarted Arnoldi on the traceless one-block propagator ``P``.

    Parameters
    ----------
    Ham : dynamiqs operator
        Hamiltonian (possibly time dependent) of the master equation.
    jump_ops : list
        Lindblad jump operators.
    T_block : float
        Duration of one propagation block; also converts eigenvalues to rates.
    jump_ops_LdL : list, optional
        Precomputed ``L^dag L`` terms; if given, ``dq.mesolve_fast`` is used
        (needs the pinned dynamiqs fork).
    output_phase : array (N,), optional
        Diagonal unitary applied after each block as
        ``rho -> diag(phase) rho diag(phase)^dag`` (interaction-picture
        correction).
    dims : tuple of int, optional
        Per-mode Hilbert dimensions; ``N = prod(dims)``, ``dim = N*N``.
        Inferred from ``Ham.dims`` if omitted.
    rtol, atol : float
        Integrator tolerances used by the Arnoldi loop. Change them later with
        :meth:`set_tolerance`; :meth:`residual_check` can use its own.
    require_gpu : bool
        Raise if JAX has no GPU device.
    """

    def __init__(
        self,
        Ham,
        jump_ops,
        T_block,
        jump_ops_LdL=None,
        output_phase=None,
        dims=None,
        rtol=1e-9,
        atol=1e-10,
        require_gpu=True,
    ):
        if require_gpu and not any(d.platform == "gpu" for d in jax.devices()):
            raise RuntimeError("JAX is not using a GPU.")

        if dims is None:
            try:
                dims = tuple(Ham.dims)
            except AttributeError as exc:
                raise ValueError(
                    "`dims` not provided and not inferable from `Ham`; pass "
                    "e.g. dims=(n_a, n_b)."
                ) from exc
        self.dims = tuple(int(d) for d in dims)
        self.N = int(np.prod(self.dims))
        self.dim = self.N * self.N

        self.Ham = Ham
        self.jump_ops = jump_ops
        self.jump_ops_LdL = jump_ops_LdL
        self.output_phase = (
            None if output_phase is None else jnp.asarray(output_phase, dtype=COMPLEX)
        )
        self.T_block = T_block
        self.tsave = jnp.array([0.0, T_block], dtype=REAL)

        self._has_lindbladian = True
        self._propagators = {}  # (rtol, atol) -> jitted propagator
        self._build_helpers()
        self.set_tolerance(rtol, atol)

    @classmethod
    def from_operator(cls, apply, dim, T_block=1.0):
        """Build a solver around an arbitrary linear map ``apply(v) -> v``.

        No Lindbladian, no trace projection, no Hermitization: the Krylov
        vectors are plain vectors of length ``dim``. This is what the CPU test
        in ``tests/test_krylov_schur_cpu.py`` uses to check the restart algebra
        against a dense matrix. :meth:`set_tolerance` is unavailable.
        """
        self = cls.__new__(cls)
        self.dims = None
        self.N = None
        self.dim = int(dim)
        self.Ham = self.jump_ops = self.jump_ops_LdL = self.output_phase = None
        self.T_block = T_block
        self.tsave = None
        self._has_lindbladian = False
        self._propagators = {}
        self.rtol = self.atol = None
        self._build_helpers()
        self.propagate = jax.jit(apply)
        self._build_expand()
        return self

    # ------------------------------------------------------------------ #
    # Vectorization + linear-algebra helpers                              #
    # ------------------------------------------------------------------ #
    def _build_helpers(self):
        N = self.N

        @jax.jit
        def normalize(v):
            return v / jnp.linalg.norm(v)

        self.normalize = normalize

        if N is None:  # generic operator: no density-matrix structure
            self.vec = self.unvec = None
            self.id_vec = None
            self.project_traceless = jax.jit(lambda v: v)
            self.hermitize = jax.jit(lambda v: v)
            return

        def vec(M):
            return M.T.reshape(-1)

        def unvec(v):
            return v.reshape(N, N).T

        id_vec = vec(jnp.eye(N, dtype=COMPLEX))

        @jax.jit
        def project_traceless(v):
            return v - (jnp.vdot(id_vec, v) / N) * id_vec

        @jax.jit
        def hermitize(v):
            M = unvec(v)
            return vec(0.5 * (M + M.conj().T))

        self.vec, self.unvec = vec, unvec
        self.id_vec = id_vec
        self.project_traceless = project_traceless
        self.hermitize = hermitize

    def make_x0(self, seed=0):
        """Random Hermitian traceless seed (Ginibre matrix, hermitized)."""
        rng = np.random.default_rng(seed)
        x0 = rng.normal(size=self.dim) + 1j * rng.normal(size=self.dim)
        x0 = jnp.asarray(x0, dtype=COMPLEX)
        return self.project_traceless(self.hermitize(x0))

    # ------------------------------------------------------------------ #
    # Propagator P restricted to the traceless subspace                   #
    # ------------------------------------------------------------------ #
    def _make_propagator(self, rtol, atol):
        """Jitted one-block propagator at the given integrator tolerances."""
        if not self._has_lindbladian:
            raise RuntimeError(
                "This solver wraps a generic operator (from_operator); it has "
                "no integrator tolerance to set."
            )
        dims = self.dims
        Ham, jump_ops, jump_ops_LdL = self.Ham, self.jump_ops, self.jump_ops_LdL
        tsave = self.tsave
        method = dq.method.Tsit5(rtol=rtol, atol=atol)
        vec, unvec = self.vec, self.unvec
        project = self.project_traceless
        phase = self.output_phase

        @jax.jit
        def propagate(rho_vec):
            rho_vec = project(rho_vec)
            rho0 = dq.asqarray(unvec(rho_vec), dims=dims)
            # `assume_hermitian` must be False: after the first restart the
            # Krylov vectors are complex combinations and no longer Hermitian.
            if jump_ops_LdL is not None:
                res = dq.mesolve_fast(
                    Ham,
                    jump_ops,
                    jump_ops_LdL,
                    rho0,
                    tsave,
                    method=method,
                    assume_hermitian=False,
                )
            else:
                res = dq.mesolve(
                    Ham,
                    jump_ops,
                    rho0,
                    tsave,
                    method=method,
                    assume_hermitian=False,
                )
            rho_T = res.states[-1].to_jax()
            if phase is not None:
                rho_T = phase[:, None] * rho_T * jnp.conj(phase)[None, :]
            return project(vec(rho_T))

        return propagate

    def _get_propagator(self, rtol, atol):
        key = (float(rtol), float(atol))
        if key not in self._propagators:
            self._propagators[key] = self._make_propagator(*key)
        return self._propagators[key]

    def set_tolerance(self, rtol, atol):
        """Switch the Arnoldi loop to new integrator tolerances.

        Propagators are cached per ``(rtol, atol)``, so toggling between two
        settings compiles each once. A Krylov-Schur ``state`` produced at one
        tolerance can be resumed at another: the kept subspace is still a good
        subspace, and the residual re-adjusts within the first cycle.
        """
        self.rtol, self.atol = float(rtol), float(atol)
        self.propagate = self._get_propagator(rtol, atol)
        self._build_expand()

    # ------------------------------------------------------------------ #
    # Arnoldi expansion of a Krylov-Schur decomposition (jitted)          #
    # ------------------------------------------------------------------ #
    def _build_expand(self):
        propagate = self.propagate
        project = self.project_traceless

        @functools.partial(jax.jit, static_argnames=("k", "m"))
        def expand(Q, B, k, m):
            """Grow the decomposition from order ``k`` to order ``m``.

            On entry rows ``0..k`` of ``Q`` are filled (the ``k`` kept vectors
            plus the residual direction) and ``B[:k+1, :k]`` holds the kept
            block and its coupling row; everything else is zero. ``k = 0`` is
            a plain Arnoldi start from ``Q[0]``.
            """

            def body(j, state):
                Q, B = state
                v = propagate(Q[j])
                # Two-pass classical Gram-Schmidt. Each pass is one GEMV against
                # the whole basis; rows beyond j are zero and contribute nothing.
                h1 = Q.conj() @ v
                v = v - h1 @ Q
                h2 = Q.conj() @ v
                v = v - h2 @ Q
                h = h1 + h2
                v = project(v)
                beta = jnp.linalg.norm(v)
                B = B.at[:, j].set(h)
                B = B.at[j + 1, j].set(beta)
                Q = Q.at[j + 1].set(v / beta)
                return Q, B

            return jax.lax.fori_loop(k, m, body, (Q, B))

        self._expand = expand

    # ------------------------------------------------------------------ #
    # Small dense algebra on the host                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def select_target(theta, target="LR"):
        """Index of the target Ritz value: largest real part, or nearest ``target``."""
        theta = np.asarray(theta)
        if isinstance(target, str):
            if target.upper() == "LR":
                return int(np.argmax(theta.real))
            if target.upper() == "LM":
                return int(np.argmax(np.abs(theta)))
            raise ValueError(f"unknown target {target!r}; use 'LR', 'LM' or a complex number")
        return int(np.argmin(np.abs(theta - complex(target))))

    @staticmethod
    def ritz_pairs(B, m):
        """Ritz values, eigenvectors and residual norms of the order-``m`` decomposition.

        Returns ``(theta, Z, res)``: ``theta[i]`` and unit-norm ``Z[:, i]`` are
        the eigenpairs of ``B[:m, :m]``; ``res[i] = |B[m, :] @ Z[:, i]|`` is the
        exact residual norm ``||P y_i - theta_i y_i||`` of the Ritz vector
        ``y_i = Q[:m].T @ Z[:, i]`` (which has unit norm).
        """
        Bmm = np.asarray(B[:m, :m])
        b = np.asarray(B[m, :m])
        theta, Z = np.linalg.eig(Bmm)
        res = np.abs(b @ Z)
        return theta, Z, res

    def restart(self, Q, B, m, k, idx_target, theta, Z):
        """Truncate the order-``m`` decomposition to the ``k`` Ritz vectors kept.

        Kept: the target, then the ``k - 1`` remaining Ritz values of largest
        modulus (the slow cluster and the outermost bulk). The kept eigenvectors
        of ``B[:m, :m]`` are orthonormalized (QR) into ``W``; because they span
        an invariant subspace, ``P (Q[:m].T W) = (Q[:m].T W) S + Q[m] b^T`` with
        ``S = W^H B W`` and ``b = B[m, :] @ W``, up to a defect that is returned
        for monitoring. This is Stewart's Krylov-Schur restart with Ritz vectors
        in place of Schur vectors (Morgan's thick restart).

        A complex-conjugate pair has exactly equal modulus, so it can only be
        split by the ``k`` cut at the very last slot; when that happens the cut
        is extended by one rather than keeping half a pair. The number actually
        kept is ``info["k_kept"]``, which the caller must use as the new order.

        Returns ``(Q_new, B_new, info)`` with the same array shapes.
        """
        Bmm = np.asarray(B[:m, :m])
        b = np.asarray(B[m, :m])

        order = [idx_target] + [
            int(i) for i in np.argsort(-np.abs(theta)) if i != idx_target
        ]
        keep = list(order[:k])
        if k + 1 < m:  # do not split a conjugate pair at the cut (and keep k < m)
            nxt = order[k]
            if any(abs(theta[nxt] - np.conj(theta[j])) < 1e-12 * max(1.0, abs(theta[nxt]))
                   and abs(theta[nxt].imag) > 0 for j in keep):
                keep.append(nxt)
        keep = np.array(keep)
        k = len(keep)
        W, _ = np.linalg.qr(Z[:, keep])  # (m, k), orthonormal columns
        S = W.conj().T @ Bmm @ W  # (k, k)
        b_new = b @ W  # (k,)
        defect = float(np.linalg.norm(Bmm @ W - W @ S))

        Wj = jnp.asarray(W, dtype=COMPLEX)
        Q_new = jnp.zeros_like(Q)
        Q_new = Q_new.at[:k].set(Wj.T @ Q[:m])
        Q_new = Q_new.at[k].set(Q[m])
        B_new = jnp.zeros_like(B)
        B_new = B_new.at[:k, :k].set(jnp.asarray(S, dtype=COMPLEX))
        B_new = B_new.at[k, :k].set(jnp.asarray(b_new, dtype=COMPLEX))

        info = {
            "kept": keep,
            "k_kept": k,
            "theta_kept": theta[keep],
            "defect": defect,
        }
        return Q_new, B_new, info

    # ------------------------------------------------------------------ #
    # The restarted iteration                                             #
    # ------------------------------------------------------------------ #
    def solve(
        self,
        x0=None,
        *,
        m=40,
        k=10,
        res_tol=1e-8,
        max_cycles=20,
        target="LR",
        state=None,
        hermitize_ritz=True,
        log=print,
    ):
        """Run thick-restarted Arnoldi until the target residual is below ``res_tol``.

        Parameters
        ----------
        x0 : array, optional
            Starting vector (cold start). Ignored when ``state`` is given.
        m : int
            Basis size at the end of a cycle (Krylov vectors in memory: ``m+1``).
        k : int
            Vectors kept at a restart: the target plus ``k - 1`` neighbours.
            ``k = 1`` is the single-vector explicit restart, kept for comparison.
        res_tol : float
            Stop when ``||P y - theta y|| / |theta| <= res_tol`` for the target.
        max_cycles : int
            Cycle budget. Each cycle costs ``m - k`` applications (``m`` for the
            first cold one).
        target : "LR", "LM" or complex
            Which Ritz value is the target (see :meth:`select_target`).
        state : dict, optional
            ``out["state"]`` of a previous call, to resume (e.g. after
            :meth:`set_tolerance`). Must have been produced with ``k < m``.
        hermitize_ritz : bool
            Hermitize the returned Ritz vector (right for a real target).
        log : callable or None
            Per-cycle progress line.

        Returns
        -------
        dict
            ``mu`` (target Ritz value), ``res_rel``, ``x_ritz`` (unit norm),
            ``converged``, ``n_apply``, ``n_cycles``, ``history`` (one dict per
            cycle, plain numpy), ``ritz`` (all Ritz values / residuals of the
            final cycle) and ``state`` (truncated decomposition, to resume).
        """
        if k < 1 or k >= m:
            raise ValueError(f"need 1 <= k < m, got k={k}, m={m}")
        if max_cycles < 1:
            raise ValueError(f"max_cycles must be >= 1, got {max_cycles}")
        dim = self.dim
        log = log or (lambda *_: None)

        Q = jnp.zeros((m + 1, dim), dtype=COMPLEX)
        B = jnp.zeros((m + 1, m), dtype=COMPLEX)
        if state is not None:
            k_cur = int(state["k"])
            if k_cur >= m:
                raise ValueError(f"state keeps k={k_cur} vectors, need k < m={m}")
            Q = Q.at[: k_cur + 1].set(jnp.asarray(state["Q"])[: k_cur + 1])
            B = B.at[: k_cur + 1, :k_cur].set(jnp.asarray(state["B"])[: k_cur + 1, :k_cur])
            n_apply = int(state.get("n_apply", 0))
        else:
            if x0 is None:
                raise ValueError("give x0 for a cold start or state to resume")
            Q = Q.at[0].set(self.normalize(self.project_traceless(jnp.asarray(x0, dtype=COMPLEX))))
            k_cur = 0
            n_apply = 0

        history = []
        converged = False
        t_start = time.time()
        for cycle in range(max_cycles):
            t0 = time.time()
            Q, B = self._expand(Q, B, k=k_cur, m=m)
            B.block_until_ready()
            n_apply += m - k_cur
            t_expand = time.time() - t0

            theta, Z, res = self.ritz_pairs(B, m)
            idx = self.select_target(theta, target)
            res_rel = float(res[idx] / abs(theta[idx]))
            converged = res_rel <= res_tol
            last = converged or cycle == max_cycles - 1

            info = None
            if not last:
                Q, B, info = self.restart(Q, B, m, k, idx, theta, Z)
                k_cur = info["k_kept"]  # may be k+1, see `restart`

            entry = {
                "cycle": cycle,
                "n_apply": n_apply,
                "mu": complex(theta[idx]),
                "res_rel": res_rel,
                "theta": np.asarray(theta),
                "res": np.asarray(res),
                "theta_kept": None if info is None else np.asarray(info["theta_kept"]),
                "defect": None if info is None else info["defect"],
                "t_cycle": time.time() - t0,
                "t_expand": t_expand,
                "rtol": self.rtol,
                "atol": self.atol,
            }
            history.append(entry)
            kept_str = (
                ""
                if info is None
                else " | kept |theta|: "
                + " ".join(f"{a:.4f}" for a in np.abs(info["theta_kept"]))
            )
            log(
                f"cycle {cycle:3d} | apps {n_apply:5d} | "
                f"1-Re mu = {1 - theta[idx].real: .3e} | Im mu = {theta[idx].imag: .1e} | "
                f"res_rel = {res_rel:.3e} | {t_expand:6.1f} s{kept_str}"
            )
            if converged:
                break

        # Ritz vector of the target from the (untruncated) final decomposition.
        x_ritz = Q[:m].T @ jnp.asarray(Z[:, idx], dtype=COMPLEX)
        if hermitize_ritz:
            x_ritz = self.hermitize(x_ritz)
        x_ritz = self.normalize(self.project_traceless(x_ritz))

        # Resumable state: the same truncation a further cycle would start from.
        Q_s, B_s, info_s = self.restart(Q, B, m, k, idx, theta, Z)
        state_out = {
            "Q": Q_s,
            "B": B_s,
            "k": info_s["k_kept"],
            "m": m,
            "n_apply": n_apply,
        }

        return {
            "mu": complex(theta[idx]),
            "res_rel": res_rel,
            "x_ritz": x_ritz,
            "converged": bool(converged),
            "n_apply": int(n_apply),
            "n_cycles": len(history),
            "t_total": time.time() - t_start,
            "history": history,
            "ritz": {"theta": np.asarray(theta), "res": np.asarray(res), "idx_target": idx},
            "state": state_out,
        }

    # ------------------------------------------------------------------ #
    # Ritz vector, residual, rate                                         #
    # ------------------------------------------------------------------ #
    def ritz_vector(self, Q, B, m, target="LR", hermitize=True):
        """Ritz vector of ``P`` for the target Ritz value of an order-``m`` decomposition.

        Same contract as ``ArnoldiLindblad.ritz_vector``; returns
        ``(x_ritz, rho_ritz)``. ``rho_ritz`` is ``None`` for a generic operator.
        """
        theta, Z, _ = self.ritz_pairs(B, m)
        idx = self.select_target(theta, target)
        x = Q[:m].T @ jnp.asarray(Z[:, idx], dtype=COMPLEX)
        if hermitize:
            x = self.hermitize(x)
        x = self.normalize(self.project_traceless(x))
        rho = None if self.N is None else dq.asqarray(self.unvec(x), dims=self.dims)
        return x, rho

    def residual_check(self, x_ritz, n_blocks=1, rtol=None, atol=None):
        """Rayleigh quotient and relative residual of ``x_ritz`` on ``P**n_blocks``.

        ``P`` is applied ``n_blocks`` times in a row (with the trace projection
        in between, which ``P`` preserves anyway). For a small rate this is
        what makes ``1 - mu`` resolvable above the integrator tolerance: the
        rate is ``-log(mu_n) / (n_blocks * T_block)``. Pass ``rtol``/``atol`` to
        use a tighter integrator than the loop's (compiled once, cached).
        """
        if rtol is None or atol is None:
            propagate = self.propagate
        else:
            propagate = self._get_propagator(rtol, atol)
        x = jnp.asarray(x_ritz, dtype=COMPLEX)
        Px = x
        for _ in range(int(n_blocks)):
            Px = propagate(Px)
        mu = jnp.vdot(x, Px) / jnp.vdot(x, x)
        res_rel = jnp.linalg.norm(Px - mu * x) / (jnp.abs(mu) * jnp.linalg.norm(x))
        return {
            "mu_RQ": mu,
            "res_rel": res_rel,
            "rate_RQ": self.rate_from_mu(mu, n_blocks=n_blocks),
            "n_blocks": int(n_blocks),
        }

    def rate_from_mu(self, mu, n_blocks=1):
        """Decay rate ``-log(mu) / (n_blocks * T_block)``."""
        return -jnp.log(mu) / (n_blocks * self.T_block)
