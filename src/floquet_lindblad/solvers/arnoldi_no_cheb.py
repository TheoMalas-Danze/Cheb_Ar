"""Plain Arnoldi iteration on a vectorized Lindbladian (Floquet) propagator.

``ArnoldiLindblad`` is the Chebyshev-free counterpart of
:class:`~floquet_lindblad.solvers.cheb_ar.ChebAr`: same target (the slowest decaying
eigenvalue of the one-block propagator ``P``), same interaction-frame hooks
(``jump_ops_LdL``, ``output_phase``), but no spectral filter — it just runs
Arnoldi to convergence. Useful as a reference check on ``ChebAr``'s filtered
result, and as a fallback when the ellipse fit misbehaves.

Like ``ChebAr`` it is model-agnostic: it takes a Lindbladian (Hamiltonian,
possibly time dependent, plus jump operators) and iterates on ``P`` restricted
to the traceless subspace. Restricting to traceless matrices deflates the steady
state exactly, so the target eigenvalue is the dominant eigenvalue of the
restricted map.

Conventions
-----------
* Column-major vectorization everywhere: ``vec(M) = M.T.reshape(-1)`` and
  ``unvec(v) = v.reshape(N, N).T`` (matches ``dq.vectorize``). Use the solver's
  own ``unvec`` to rebuild a density matrix -- a bare ``reshape(N, N)`` gives
  the transpose, i.e. the complex conjugate for Hermitian input.
* Arnoldi basis ``Q`` has shape ``(m+1, dim)``; row ``k`` is the ``k``-th
  basis vector. ``H`` has shape ``(m+1, m)``.
* All Krylov vectors are Hermitian and traceless (Hermitian seed, ``P``
  preserves both), so ``H`` is real up to integrator noise:
  ``np.abs(H.imag).max()`` is a free estimate of the noise floor.

Typical use
-----------
>>> solver = ArnoldiLindblad(H, jump_ops, T_block, dims=(20, 8))
>>> x0 = solver.make_x0(seed=0)
>>> Q, H, steps, mu_list = solver.arnoldi(x0, m_new=60)
>>> Q, H, steps, mu_list = solver.arnoldi(  # extend the same factorization
...     x0, m_new=100, m_old=60, Q_old=Q, H_old=H
... )
>>> x_ritz, rho_ritz = solver.ritz_vector(Q, H, 100, target=mu_list[-1])
>>> solver.residual_check(x_ritz)
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import dynamiqs as dq

dq.set_precision("double")

COMPLEX = jnp.complex128
REAL = jnp.float64


class ArnoldiLindblad:
    """Arnoldi iteration on the traceless one-block propagator ``P``.

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
        Integrator tolerances.
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
        self.method = dq.method.Tsit5(rtol=rtol, atol=atol)

        self._build_helpers()
        self._build_propagator()
        self._build_arnoldi()

    # ------------------------------------------------------------------ #
    # Vectorization + linear-algebra helpers                              #
    # ------------------------------------------------------------------ #
    def _build_helpers(self):
        N = self.N

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

        @jax.jit
        def normalize(v):
            return v / jnp.linalg.norm(v)

        self.vec, self.unvec = vec, unvec
        self.id_vec = id_vec
        self.project_traceless = project_traceless
        self.hermitize = hermitize
        self.normalize = normalize

    def make_x0(self, seed=0):
        """Random Hermitian traceless seed (Ginibre matrix, hermitized)."""
        rng = np.random.default_rng(seed)
        x0 = rng.normal(size=self.dim) + 1j * rng.normal(size=self.dim)
        x0 = jnp.asarray(x0, dtype=COMPLEX)
        return self.project_traceless(self.hermitize(x0))

    # ------------------------------------------------------------------ #
    # Propagator P restricted to the traceless subspace                   #
    # ------------------------------------------------------------------ #
    def _build_propagator(self):
        dims = self.dims
        Ham, jump_ops, jump_ops_LdL = self.Ham, self.jump_ops, self.jump_ops_LdL
        tsave, method = self.tsave, self.method
        vec, unvec = self.vec, self.unvec
        project = self.project_traceless
        phase = self.output_phase

        @jax.jit
        def propagate(rho_vec):
            rho_vec = project(rho_vec)
            rho0 = dq.asqarray(unvec(rho_vec), dims=dims)
            # `assume_hermitian` is a flat keyword in the pinned dynamiqs fork,
            # not a member of an `options=dq.Options(...)` object -- neither
            # entry point accepts `options`. It must be False here: the Krylov
            # vectors are Hermitian, but the integrator is handed intermediate
            # states that need not be.
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

        self.propagate = propagate

    # ------------------------------------------------------------------ #
    # Arnoldi factorization (cold start or extension)                     #
    # ------------------------------------------------------------------ #
    def _build_arnoldi(self):
        dim = self.dim
        propagate = self.propagate
        normalize = self.normalize
        project = self.project_traceless

        @functools.partial(jax.jit, static_argnames=("m_new", "m_old"))
        def arnoldi_factorization(x0, m_new, m_old=0, Q_old=None, H_old=None):
            Q = jnp.zeros((m_new + 1, dim), dtype=COMPLEX)
            H = jnp.zeros((m_new + 1, m_new), dtype=COMPLEX)

            if m_old == 0:
                Q = Q.at[0].set(normalize(project(x0)))
            else:
                Q = Q.at[: m_old + 1].set(Q_old)
                H = H.at[: m_old + 1, :m_old].set(H_old)

            def body(k, state):
                Q, H = state
                v = propagate(Q[k])

                # modified Gram-Schmidt, two passes
                def mgs(j, inner):
                    v, H = inner
                    h = jnp.vdot(Q[j], v)
                    return v - h * Q[j], H.at[j, k].add(h)

                v, H = jax.lax.fori_loop(0, k + 1, mgs, (v, H))
                v, H = jax.lax.fori_loop(0, k + 1, mgs, (v, H))

                beta = jnp.linalg.norm(v)
                H = H.at[k + 1, k].set(beta)
                Q = Q.at[k + 1].set(v / beta)
                return Q, H

            Q, H = jax.lax.fori_loop(m_old, m_new, body, (Q, H))
            return Q, H

        self._arnoldi_factorization = arnoldi_factorization

    @staticmethod
    def track_target(H, m_new, m_old=0, mu_ref=1.0, stride=1):
        """Ritz value nearest ``mu_ref`` after k steps, for k in [m_old, m_new].

        Returns ``(steps, mu)``, both 1-d arrays of the same length: the step
        counts actually sampled (``stride`` apart, always including ``m_new``)
        and the tracked Ritz value at each.
        """
        H = np.asarray(H)
        steps = list(range(m_old + 1, m_new + 1, stride))
        if not steps:
            steps = [m_new]
        elif steps[-1] != m_new:
            steps.append(m_new)
        out = []
        for k in steps:
            ritz = np.linalg.eigvals(H[:k, :k])
            out.append(ritz[np.argmin(np.abs(ritz - mu_ref))])
        return np.array(steps), np.array(out)

    def arnoldi(
        self, x0, m_new, *, m_old=0, Q_old=None, H_old=None, mu_ref=1.0, stride=1
    ):
        """Run (or extend) the factorization and track the target Ritz value.

        Returns ``(Q, H, steps, mu_list)``; ``steps`` and ``mu_list`` line up
        elementwise, so plot against ``steps``, not ``range(m_new)``.
        """
        if m_old and (Q_old is None or H_old is None):
            raise ValueError("Extension requires Q_old and H_old.")
        Q, H = self._arnoldi_factorization(
            x0, m_new, m_old=m_old, Q_old=Q_old, H_old=H_old
        )
        steps, mu_list = self.track_target(
            H, m_new, m_old=m_old, mu_ref=mu_ref, stride=stride
        )
        return Q, H, steps, mu_list

    # ------------------------------------------------------------------ #
    # Ritz vector, residual, rate                                         #
    # ------------------------------------------------------------------ #
    def ritz_vector(self, Q, H, m, target):
        """Ritz vector of ``P`` whose Ritz value is nearest ``target``.

        ``m`` must be the step count the factorization actually reached (the
        last entry of ``steps``); a smaller value silently throws away Krylov
        vectors and gives a worse eigenvector than ``mu_list[-1]`` suggests.

        Returns ``(x_ritz, rho_ritz)``: the normalized vectorized matrix and
        the corresponding ``dq`` array.
        """
        Hm = np.asarray(H[:m, :m])
        ritz_vals, ritz_vecs = np.linalg.eig(Hm)
        idx = int(np.argmin(np.abs(ritz_vals - target)))
        y = jnp.asarray(ritz_vecs[:, idx], dtype=COMPLEX)

        x = Q[:m].T @ y
        x = self.normalize(self.project_traceless(self.hermitize(x)))
        rho = dq.asqarray(self.unvec(x), dims=self.dims)
        return x, rho

    def residual_check(self, x_ritz):
        """Rayleigh quotient and relative residual on the true ``P``."""
        Px = self.propagate(x_ritz)
        mu = jnp.vdot(x_ritz, Px) / jnp.vdot(x_ritz, x_ritz)
        res_rel = jnp.linalg.norm(Px - mu * x_ritz) / jnp.abs(mu)
        return {"mu_RQ": mu, "res_rel": res_rel, "rate_RQ": self.rate_from_mu(mu)}

    def rate_from_mu(self, mu):
        """Decay rate ``-log(mu) / T_block``."""
        return -jnp.log(mu) / self.T_block
