"""Chebyshev-Arnoldi iteration for a vectorized Lindbladian propagator.

This module wraps the pipeline of ``full_ATS/Chebyshev-Arnoldi.ipynb``.

The ``ChebAr`` class is *model-agnostic*: it takes the Lindbladian
(Hamiltonian ``H`` + jump operators) directly and works on the projected
one-period propagator ``P`` of the associated master equation. The full ATS
Hamiltonian is built by the standalone :func:`build_ats_hamiltonian` so that
the model and the solver stay decoupled.

Typical use
-----------
>>> from cheb_ar import ChebAr
>>> # model builders (e.g. the ATS Hamiltonian) live outside the solver;
>>> # they will be collected in ``cheb_ar.models``
>>> H, jump_ops, T_block, _ = build_ats_hamiltonian(alpha_sq=8.5)
>>> solver = ChebAr(H, jump_ops, T_block, dims=(20, 11), cheb_degree=6)
>>> x0 = solver.make_x0(seed=0)
>>>
>>> # 1. first estimation: plain Arnoldi of P (no filtering)
>>> Q0, H0, ritz_vals = solver.first_estimation(x0, m_arnoldi=40)
>>>
>>> # 2. set up the Chebyshev filter from the estimated spectrum
>>> solver.setup_chebyshev(ritz_vals, margin=4e-2)
>>>
>>> # 3. filtered Arnoldi (warm-restartable); mu_list[-1] is the bit-flip eig
>>> Q, H, mu_list = solver.arnoldi_hessenberg(x0, solver.chebyshev_filter, 40)
>>> Q, H, mu_list = solver.arnoldi_hessenberg(
...     x0, solver.chebyshev_filter, 80,
...     m_old=40, Q_old=Q, H_old=H, mu_list_old=mu_list)
>>> rate = solver.rate_from_mu(mu_list[-1])
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs
from scipy.special import jv

jax.config.update("jax_enable_x64", True)

import dynamiqs as dq

dq.set_precision("double")

# Dtypes used throughout (double precision).
WANTED_TYPE_COMPLEX = jnp.complex128
WANTED_TYPE_REAL = jnp.float64

class ChebAr:
    """Chebyshev-Arnoldi iteration on a vectorized Lindbladian propagator.

    The solver builds the projected one-period propagator ``P`` of the master
    equation defined by ``(Ham, jump_ops)`` and runs the Chebyshev-accelerated
    Arnoldi iteration on it.

    Parameters
    ----------
    Ham : dynamiqs operator
        Hamiltonian (possibly time dependent) of the master equation.
    jump_ops : list
        Lindblad jump operators.
    T_block : float
        Duration of one propagation block (sets ``tsave = [0, T_block]`` and
        converts eigenvalues to rates).
    dims : tuple of int, optional
        Per-mode Hilbert-space dimensions of the composite system (Fock
        truncations, spin multiplicities, ...). The total Hilbert dimension
        is ``N = prod(dims)`` and the Liouville (vectorized) dimension is
        ``dim = N*N``. This is the only piece of system-specific information
        ``ChebAr`` needs, and it supports an arbitrary number of subsystems
        (not just the two-mode ``(n_a, n_b)`` case). If omitted, ``ChebAr``
        tries to read ``Ham.dims`` itself (this works whenever ``Ham`` was
        built with ``dq.tensor``/``dq.asqarray`` and carries that metadata).
    cheb_degree : int
        Degree of the Chebyshev filtering polynomial.
    rtol, atol : float
        ``mesolve`` integration tolerances for the jitted (jax) propagator.
    require_gpu : bool
        If ``True`` (default, matching the notebook), raise ``RuntimeError`` at
        construction unless JAX has a GPU device. Set ``False`` to allow a CPU
        fallback (much slower).
    """

    def __init__(
        self,
        Ham,
        jump_ops,
        T_block,
        jump_ops_LdL=None,
        output_phase=None,
        dims=None,
        cheb_degree=6,
        rtol=1e-9,
        atol=1e-10,
        require_gpu=True,
    ):
        # match the notebook: fail fast if JAX is not on a GPU
        if require_gpu and not any(d.platform == "gpu" for d in jax.devices()):
            raise RuntimeError("JAX is not using a GPU.")

        # --- Hilbert-space bookkeeping: the only system-dependent input ---
        if dims is None:
            try:
                dims = tuple(Ham.dims)
            except AttributeError as exc:
                raise ValueError(
                    "`dims` was not provided and could not be inferred from "
                    "`Ham` (no `.dims` attribute). Pass the per-mode "
                    "Hilbert-space dimensions explicitly, "
                    "e.g. dims=(n_a, n_b) or dims=(n,) for a single mode."
                ) from exc
        self.dims = tuple(int(d) for d in dims)
        self.N = int(np.prod(self.dims))
        self.dim = self.N * self.N

        # --- Lindbladian ---
        self.Ham = Ham
        self.jump_ops = jump_ops
        self.jump_ops_LdL = jump_ops_LdL
        self.output_phase = output_phase
        self.cheb_degree = cheb_degree
        self.T_block = T_block
        self.tsave = jnp.array([0.0, T_block], dtype=WANTED_TYPE_REAL)

        # integrator
        self.method = dq.method.Tsit5(rtol=rtol, atol=atol)

        self._build_helpers()
        self._build_propagator()

        # filled in once the ellipse / Chebyshev filter are set up
        self.ellipse = None
        self.chebyshev_filter = None
        self._arnoldi_hessenberg = None

    # ------------------------------------------------------------------ #
    # Linear-algebra helpers (jitted)                                     #
    # ------------------------------------------------------------------ #
    def _build_helpers(self):
        """Build the jitted ``normalize`` and ``project_trace_zero_vec``."""
        NN = self.N
        # vectorized identity, used to project out the trace
        id_vec = dq.vectorize(dq.eye(NN)).to_jax().reshape(-1)
        self.id_vec = id_vec

        @jax.jit
        def normalize(x):
            return x / jnp.linalg.norm(x)

        @jax.jit
        def project_trace_zero_vec(rho_vec):
            tr = jnp.vdot(id_vec, rho_vec)
            return rho_vec - (tr / NN) * id_vec

        self.normalize = normalize
        self.project_trace_zero_vec = project_trace_zero_vec

    @staticmethod
    @jax.jit
    def hermitize_vec(x0):
        """Project a vectorized operator onto its Hermitian part.

        ``vec(rho) -> vec((rho + rho^dag) / 2)``. ``x0`` is a flat array of
        shape ``(N*N,)``.
        """
        N = round(np.sqrt(x0.shape[-1]))
        M = x0.reshape(N, N)
        M = 0.5 * (M + M.conj().T)
        return M.reshape(-1)

    def make_x0(self, seed=0, hermitize=True):
        """Build a random complex starting vector (optionally Hermitized)."""
        rng = np.random.default_rng(seed)
        x0 = rng.normal(size=self.dim) + 1j * rng.normal(size=self.dim)
        x0 = jnp.array(x0, dtype=WANTED_TYPE_COMPLEX)
        if hermitize:
            x0 = self.hermitize_vec(x0)
        return x0

    # ------------------------------------------------------------------ #
    # Projected one-period propagator P (jitted, jax/GPU)                 #
    # ------------------------------------------------------------------ #
    def _build_propagator(self):
        """Build the jitted projected propagator ``propagate_block_projected``."""
        dims = self.dims
        Ham = self.Ham
        jump_ops = self.jump_ops
        jump_ops_LdL = self.jump_ops_LdL
        tsave = self.tsave
        method = self.method
        project = self.project_trace_zero_vec
        phase = self.output_phase

        @jax.jit
        def propagate_block_projected(rho_vec):
            rho_vec = project(rho_vec)
            rho_flat = dq.unvectorize(dq.asqarray(rho_vec[:, None]))
            rho_0 = dq.asqarray(dq.to_jax(rho_flat), dims=dims)
            if jump_ops_LdL is not None:
                res = dq.mesolve_fast(
                    Ham,
                    jump_ops,
                    jump_ops_LdL,
                    rho_0,
                    tsave,
                    method=method,
                    options=dq.Options(assume_hermitian=False),
                )
            else:
                res = dq.mesolve(
                        Ham,
                        jump_ops,
                        rho_0,
                        tsave,
                        method=method,
                        options=dq.Options(assume_hermitian=False),
                    )
            rho_final = res.states[-1].to_jax()
            if phase is not None:
                rho_final = phase[:, None] * rho_final * jnp.conj(phase)[None, :]
            rho_vec_final = rho_final.T.reshape(-1)
            return project(rho_vec_final)

        self.propagate_block_projected = propagate_block_projected

    # ------------------------------------------------------------------ #
    # scipy reference: dominant eigenvalue of the (numpy) propagator      #
    # ------------------------------------------------------------------ #
    def scipy_dominant_eig(self, x0=None, k=1, which="LR"):
        """Reference dominant eigenpair of ``P`` via ``scipy.sparse.linalg.eigs``.

        Uses a numpy (CPU) propagator with column-major vectorization,
        mirroring the notebook. Returns ``(vals, vecs)``.
        """
        dims = self.dims
        N = self.N
        Ham = self.Ham
        jump_ops = self.jump_ops
        tsave = self.tsave

        def mat_to_vec(rho_mat):
            return rho_mat.reshape(-1, order="F")

        def vec_to_mat(rho_vec, N):
            return rho_vec.reshape((N, N), order="F")

        id_mat = np.eye(N, dtype=np.complex64)
        id_vec = mat_to_vec(id_mat)

        def propagate_one_period_projected(rho_vec):
            rho_mat_np = vec_to_mat(rho_vec, N)
            tr = np.trace(rho_mat_np)
            rho_mat_np = rho_mat_np - (tr / N) * id_mat
            rho_mat = dq.asqarray(rho_mat_np, dims=dims)
            result = dq.mesolve(
                Ham,
                jump_ops,
                rho_mat,
                tsave,
                options=dq.Options(assume_hermitian=False),
            )
            rho_mat = (result.states[-1]).to_numpy()
            tr = np.trace(rho_mat)
            rho_vec = mat_to_vec(rho_mat)
            return rho_vec - (tr / N) * id_vec

        UF_proj = LinearOperator(
            shape=(self.dim, self.dim),
            matvec=propagate_one_period_projected,
            dtype=np.complex64,
        )

        if x0 is None:
            x0 = np.array(self.make_x0(seed=0, hermitize=False))
        else:
            x0 = np.asarray(x0)

        vals, vecs = eigs(UF_proj, k=k, which=which, v0=x0)
        return vals, vecs

    # ------------------------------------------------------------------ #
    # First estimation: plain Arnoldi factorization of P (no filtering)   #
    # ------------------------------------------------------------------ #
    @functools.partial(jax.jit, static_argnames=("self", "polynomial", "m_arnoldi"))
    def arnoldi_build_hessenberg(self, x0, polynomial, m_arnoldi):
        """Build an Arnoldi factorization of ``polynomial`` (modified GS + reorth).

        Returns ``(Q, H)`` with ``Q`` of shape ``(m_arnoldi+1, dim)`` and ``H``
        of shape ``(m_arnoldi+1, m_arnoldi)``.
        """
        dim = self.dim
        project = self.project_trace_zero_vec
        normalize = self.normalize

        x0 = project(x0)
        q0 = normalize(x0)

        Q = jnp.zeros((m_arnoldi + 1, dim), dtype=WANTED_TYPE_COMPLEX)
        H = jnp.zeros((m_arnoldi + 1, m_arnoldi), dtype=WANTED_TYPE_COMPLEX)
        Q = Q.at[0].set(q0)

        def body(k, state):
            Q, H = state
            v = polynomial(Q[k])

            # Modified Gram-Schmidt, first pass
            def gs_body(j, inner_state):
                v, H = inner_state
                qj = Q[j]
                h = jnp.vdot(qj, v)
                v = v - h * qj
                H = H.at[j, k].set(h)
                return v, H

            v, H = jax.lax.fori_loop(0, k + 1, gs_body, (v, H))

            # Reorthogonalization, second pass
            def reorth_body(j, inner_state):
                v, H = inner_state
                qj = Q[j]
                h_corr = jnp.vdot(qj, v)
                v = v - h_corr * qj
                H = H.at[j, k].add(h_corr)
                return v, H

            v, H = jax.lax.fori_loop(0, k + 1, reorth_body, (v, H))

            beta = jnp.linalg.norm(v)
            H = H.at[k + 1, k].set(beta)
            q_next = v / beta
            Q = Q.at[k + 1].set(q_next)
            return Q, H

        Q, H = jax.lax.fori_loop(0, m_arnoldi, body, (Q, H))
        return Q, H

    def first_estimation(self, x0, m_arnoldi=40):
        """Run plain Arnoldi of ``P`` and return ``(Q, H, ritz_vals)``.

        ``ritz_vals`` (eigenvalues of the small Hessenberg) are the spectrum
        estimate fed to :meth:`setup_chebyshev`.
        """
        Q, H = self.arnoldi_build_hessenberg(
            x0, self.propagate_block_projected, m_arnoldi
        )
        H_small = np.array(H[:m_arnoldi, :m_arnoldi])
        ritz_vals = np.linalg.eigvals(H_small)
        return Q, H, ritz_vals

    # ------------------------------------------------------------------ #
    # Enclosing ellipse for the Ritz values                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def find_enclosing_ellipse(ritz_vals, vertex_x, margin=0.0, num_a=2000):
        """Smallest-area ellipse through ``(vertex_x, 0)`` enclosing the Ritz values.

        The ellipse is centered on the real axis with semi-axes ``e`` (along Re)
        and ``b`` (along Im); ``(vertex_x, 0)`` is its right-hand vertex.
        Every returned point satisfies

            (Re - a)^2 / e^2 + Im^2 / b^2 <= 1 - margin,

        i.e. all Ritz values lie strictly *inside* the ellipse, a normalized
        distance ``margin`` away from its boundary in both the real and the
        imaginary direction. ``margin=0`` reproduces the tight minimal-area
        enclosing ellipse (the worst point sits exactly on the boundary).

        Returns ``(center_a, semi_re, semi_im)``, or ``(inf, inf, inf)`` if no
        ellipse through the vertex can enclose the points with that margin.
        """
        re = ritz_vals.real
        im = ritz_vals.imag
        level = 1.0 - margin
        if level <= 0.0:
            raise ValueError(f"margin={margin} must be < 1.")

        best_a, best_e, best_b = np.inf, np.inf, np.inf
        best_area = float("inf")

        # a < vertex_x so that (vertex_x, 0) is the right-hand vertex (e > 0)
        for a in np.linspace(-2, vertex_x - 1e-6, num_a):
            e = abs(vertex_x - a)
            # normalized horizontal coordinate of every point
            u = (re - a) / e
            # every point must keep horizontal clearance: u^2 < level so that
            # the level-set constraint below has a positive denominator
            if np.any(u**2 >= level):
                continue
            # smallest semi-minor axis keeping every point within the
            # (1 - margin) level set: Im^2 / b^2 <= level - u^2
            b = np.sqrt(np.max(im**2 / (level - u**2)))
            area = e * b
            if area < best_area:
                best_area = area
                best_a, best_e, best_b = a, e, b

        return best_a, best_e, best_b

    # ------------------------------------------------------------------ #
    # Set up the optimal Chebyshev filtering polynomial                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def chebyshev_complex(n, z):
        """``T_n(z)`` for complex ``z``, stable for ``|z| > 1``."""
        return np.cosh(n * np.arccosh(z + 0j))

    def setup_chebyshev(self, ritz_vals, margin=4e-2, num_a=2000):
        """Locate the enclosing ellipse and build the Chebyshev filter.

        ``ritz_vals`` are the eigenvalues of the small Hessenberg from a plain
        Arnoldi run on ``P`` (see :meth:`first_estimation`). The target Ritz
        value (largest real part) is deliberately left outside the ellipse and
        used as the target eigenvalue.

        Stores the ellipse parameters in ``self.ellipse`` and builds
        ``self.chebyshev_filter`` plus the warm-startable Arnoldi.
        """
        ritz_vals = np.asarray(ritz_vals)
        idx_target = np.argmax(ritz_vals.real)
        ritz_target = ritz_vals[idx_target]
        vertex_x = ritz_target - margin

        # Exclude the target Ritz value: it sits right of the vertex and is
        # the target eigenvalue, not meant to be enclosed.
        ritz_vals_enc = np.delete(ritz_vals, idx_target)

        # Enclose every remaining Ritz value with a `margin` interior clearance.
        center_a, semi_re, semi_im = self.find_enclosing_ellipse(
            ritz_vals_enc, vertex_x, margin=margin, num_a=num_a
        )

        if not all(np.isfinite([center_a, semi_re, semi_im])):
            raise RuntimeError(
                "find_enclosing_ellipse failed to find a valid ellipse "
                f"(center_a={center_a}, semi_re={semi_re}, semi_im={semi_im}). "
                "No ellipse through the vertex encloses all Ritz values with "
                f"margin={margin}; try a smaller margin or a different num_a."
            )

        # Verify every enclosed Ritz value really lies inside the ellipse.
        # level <= 1 means inside; level <= 1 - margin means margin clearance.
        levels = (
            ((ritz_vals_enc.real - center_a) / semi_re) ** 2
            + (ritz_vals_enc.imag / semi_im) ** 2
        )
        max_level = float(np.max(levels))
        if max_level > 1.0:
            raise RuntimeError(
                f"A Ritz value lies outside the enclosing ellipse "
                f"(max level {max_level:.6f} > 1)."
            )

        self.ellipse = {
            "center_a": center_a,
            "semi_re": semi_re,
            "semi_im": semi_im,
            "ritz_target": ritz_target,
            "vertex_x": vertex_x,
            "margin": margin,
            "max_level": max_level,
            "ritz_vals_enc": ritz_vals_enc,
            "idx_target": idx_target,
        }

        self._build_chebyshev_filter()
        return self.ellipse

    def optimal_polynomial_ellipse(self, m, semi_re, semi_im, lambda_1, degree):
        """Scalar optimal filtering polynomial for a spectrum in an ellipse.

        Returns a callable ``p(z)`` normalized so that ``p(lambda_1) = 1``.
        """
        d = np.sqrt((semi_re**2 - semi_im**2) + 0j)  # half-focal distance
        denom = self.chebyshev_complex(degree, (lambda_1 - m) / d)
        if abs(denom) < 1e-10:
            raise ValueError(
                "lambda_1 is too close to the ellipse boundary: "
                "denominator T_n((lambda_1-m)/d) is near zero."
            )

        def p(z):
            return self.chebyshev_complex(degree, (z - m) / d) / denom

        return p

    def _build_chebyshev_filter(self):
        """Build the jitted ``chebyshev_filter`` operator ``p(P)`` and Arnoldi.

        Uses the ellipse stored in ``self.ellipse``. The target eigenvalue for
        the *operator* polynomial is ``lambda_1 = 1`` (the target eigenvalue
        of the propagator sits at the edge of the unit disk).
        """
        if self.ellipse is None:
            raise RuntimeError("Call setup_chebyshev(...) before building the filter.")

        cheb_degree = self.cheb_degree
        m = self.ellipse["center_a"]
        semi_re = self.ellipse["semi_re"]
        semi_im = self.ellipse["semi_im"]
        d = np.sqrt((semi_re**2 - semi_im**2) + 0j)  # half-focal distance
        lambda_1 = 1.0  # target eigenvalue of P (edge of the unit disk)

        denom = self.chebyshev_complex(cheb_degree, (lambda_1 - m) / d)
        if abs(denom) < 1e-10:
            raise ValueError(
                "lambda_1 too close to the ellipse boundary: "
                "denominator T_n((lambda_1-m)/d) is near zero."
            )

        # store for the rate recovery in arnoldi_hessenberg
        self._m_ellipse = m
        self._d_ellipse = d
        self._lambda_1 = lambda_1
        self._denom = denom

        m_j = jnp.array(m, dtype=WANTED_TYPE_COMPLEX)
        d_j = jnp.array(d, dtype=WANTED_TYPE_COMPLEX)
        denom_j = jnp.array(denom, dtype=WANTED_TYPE_COMPLEX)

        propagate = self.propagate_block_projected
        project = self.project_trace_zero_vec

        @jax.jit
        def scaled_apply(v):
            """Apply the shifted/scaled operator ``w = (P - m I) / d`` to ``v``."""
            return (propagate(v) - m_j * v) / d_j

        @jax.jit
        def chebyshev_filter(x):
            x = project(x)
            if cheb_degree == 0:
                return x / denom_j

            t0 = x  # T_0(w) x = x
            t1 = scaled_apply(x)  # T_1(w) x = w x

            def body(k, state):
                t0, t1 = state
                # T_{k+1}(w) x = 2 w T_k(w) x - T_{k-1}(w) x
                t2 = 2.0 * scaled_apply(t1) - t0
                t2 = project(t2)  # usually stabilizes things
                return t1, t2

            t0, t1 = jax.lax.fori_loop(1, cheb_degree, body, (t0, t1))
            # normalize so that p(lambda_1) = 1
            return t1 / denom_j

        self.scaled_apply = scaled_apply
        self.chebyshev_filter = chebyshev_filter
        self._build_arnoldi_hessenberg()

    # ------------------------------------------------------------------ #
    # Arnoldi of the filtered operator (warm restartable) + mu recovery   #
    # ------------------------------------------------------------------ #
    def _build_arnoldi_hessenberg(self):
        """Build the warm-startable Arnoldi factorization of ``p(P)``.

        The Ritz values of the small Hessenberg approximate ``p(mu)``; the
        target eigenvalue ``mu`` is recovered each step by inverting the
        Chebyshev relation, and stored in ``mu_list``.
        """
        dim = self.dim
        project = self.project_trace_zero_vec
        normalize = self.normalize

        @functools.partial(jax.jit, static_argnames=("polynomial", "m_new", "m_old"))
        def arnoldi_hessenberg(
            x0,
            polynomial,
            m_new,
            *,
            m_old=0,
            Q_old=None,
            H_old=None,
        ):
            """Build or extend an Arnoldi factorization of ``polynomial``.

            Fresh start::

                arnoldi_hessenberg(x0, polynomial, m_new=40)

            Warm start::

                arnoldi_hessenberg(x0, polynomial, m_new=60,
                                   m_old=40, Q_old=Q, H_old=H)
            """
            Q = jnp.zeros((m_new + 1, dim), dtype=WANTED_TYPE_COMPLEX)
            H = jnp.zeros((m_new + 1, m_new), dtype=WANTED_TYPE_COMPLEX)

            # --- initialisation: fresh start vs warm start ---
            if m_old == 0:
                x0_proj = project(x0)
                q0 = normalize(x0_proj)
                Q = Q.at[0].set(q0)
            else:
                Q = Q.at[: m_old + 1].set(Q_old)
                H = H.at[: m_old + 1, :m_old].set(H_old)

            # --- shared body ---
            def body(k, state):
                Q, H = state
                v = polynomial(Q[k])

                def gs_body(j, inner_state):
                    v, H = inner_state
                    qj = Q[j]
                    h = jnp.vdot(qj, v)
                    v = v - h * qj
                    H = H.at[j, k].set(h)
                    return v, H

                v, H = jax.lax.fori_loop(0, k + 1, gs_body, (v, H))

                def reorth_body(j, inner_state):
                    v, H = inner_state
                    qj = Q[j]
                    h_corr = jnp.vdot(qj, v)
                    v = v - h_corr * qj
                    H = H.at[j, k].add(h_corr)
                    return v, H

                v, H = jax.lax.fori_loop(0, k + 1, reorth_body, (v, H))

                v = project(v)
                beta = jnp.linalg.norm(v)
                H = H.at[k + 1, k].set(beta)
                Q = Q.at[k + 1].set(v / beta)
                return Q, H

            Q, H = jax.lax.fori_loop(m_old, m_new, body, (Q, H))
            return Q, H

        self._arnoldi_hessenberg = arnoldi_hessenberg

    def recover_mu_list(self, H, m_new, *, m_old=0, mu_prev=None, warm_start=False):
        """Extract mu_list from H by tracking Ritz values across steps by continuity."""
        mu_list = []
        if warm_start:
            target = mu_prev  # carries the last known good value for warm starts
            for k in range(m_old, m_new):
                H_active = np.array(H[: k + 1, : k + 1])  # only the filled block
                ritz_vals = np.linalg.eigvals(H_active)
                if target is None:
                    # Cold start: argmax is acceptable once, on a small matrix
                    target = ritz_vals[np.argmax(ritz_vals.real)]
                else:
                    # Track by continuity: nearest Ritz value to previous target
                    target = ritz_vals[np.argmin(np.abs(ritz_vals - target))]
                mu = self._invert_chebyshev(target)
                mu_list.append(mu)
        else:
            for k in range(m_old, m_new):
                H_active = np.array(H[: k + 1, : k + 1])  # only the filled block
                ritz_vals = np.linalg.eigvals(H_active)
                target = ritz_vals[np.argmax(ritz_vals.real)]
                mu = self._invert_chebyshev(target)
                mu_list.append(mu)
        return np.array(mu_list)

    def _invert_chebyshev(self, pkmu):
        """Invert p_k(mu) -> mu, selecting the physical branch."""
        Tn_lambda1 = np.cosh(
            self.cheb_degree
            * np.arccosh((self._lambda_1 - self._m_ellipse) / self._d_ellipse + 0j)
        )
        C = pkmu * Tn_lambda1
        arccosh_C = np.arccosh(C + 0j)
        candidates = np.array(
            [
                self._m_ellipse
                + self._d_ellipse
                * np.cosh((arccosh_C + 2j * np.pi * j) / self.cheb_degree)
                for j in range(self.cheb_degree)
            ]
        )
        # Select candidate closest to unit disk (|mu| <= 1)
        return candidates[np.argmin(np.abs(np.abs(candidates) - 1.0))]

    def arnoldi_hessenberg(
        self,
        x0,
        polynomial,
        m_new,
        *,
        check_last_mu=False,
        unit_disk_tol=1e-6,
        m_old=0,
        Q_old=None,
        H_old=None,
        mu_list_old=None,
        warm_start=False,
    ):
        """Public wrapper around the (jitted) filtered Arnoldi factorization.

        After the factorization, checks that the recovered target eigenvalue
        ``mu_list[-1]`` lies inside the unit disk (it is an eigenvalue of a
        completely positive, trace-non-increasing propagator, so ``|mu| <= 1``).
        Raises ``ValueError`` if it falls outside by more than ``unit_disk_tol``.
        """
        if self._arnoldi_hessenberg is None:
            raise RuntimeError(
                "Chebyshev filter not built. Call setup_chebyshev(...) first."
            )

        Q, H = self._arnoldi_hessenberg(
            x0, polynomial, m_new, m_old=m_old, Q_old=Q_old, H_old=H_old
        )

        mu_prev = complex(mu_list_old[-1]) if mu_list_old is not None else None
        mu_list_new = self.recover_mu_list(
            H, m_new, m_old=m_old, mu_prev=mu_prev, warm_start=warm_start
        )

        # Reconstruct the full mu_list for the caller
        if mu_list_old is not None:
            mu_list = np.concatenate([mu_list_old, mu_list_new])
        else:
            mu_list = mu_list_new

        if check_last_mu:
            mu_final = complex(mu_list[-1])
            if abs(mu_final) > 1.0 + unit_disk_tol:
                raise ValueError(
                    f"mu_list[-1] = {mu_final} lies outside the unit disk "
                    f"(|mu| = {abs(mu_final):.6e} > 1 + {unit_disk_tol:.0e}); "
                    "the recovered eigenvalue is non-physical."
                )

        return Q, H, mu_list

    # ------------------------------------------------------------------ #
    # Ritz-vector reconstruction and residual check                      #
    # ------------------------------------------------------------------ #
    def ritz_vector(self, Q, H, m_arnoldi, which="abs", target=None):
        """Reconstruct the dominant full Ritz vector and its density matrix.

        Parameters
        ----------
        target : complex, optional
            If provided, selects the Ritz value closest to this value in the
            complex plane (continuity-based selection, consistent with
            recover_mu_list). Overrides ``which``.
        """
        dims = self.dims
        H_small = np.array(H[:m_arnoldi, :m_arnoldi])
        ritz_vals, ritz_vecs = np.linalg.eig(H_small)

        if target is not None:
            idx = int(np.argmin(np.abs(ritz_vals - target)))
        elif which == "abs":
            idx = int(np.argmax(np.abs(ritz_vals)))
        elif which == "real":
            idx = int(np.argmax(ritz_vals.real))
        else:
            idx = int(which)

        y = ritz_vecs[:, idx]
        x_ritz = Q[:m_arnoldi, :].T @ jnp.array(y, dtype=WANTED_TYPE_COMPLEX)
        x_ritz = self.project_trace_zero_vec(x_ritz)
        x_ritz = self.normalize(x_ritz)

        rho_ritz = dq.unvectorize(dq.asqarray(x_ritz[:, None]))
        rho_ritz = dq.asqarray(dq.to_jax(rho_ritz), dims=dims)
        return x_ritz, rho_ritz

    def residual_check(self, x_ritz):
        """Rayleigh-quotient eigenvalue and relative residual on the true ``P``.

        Returns a dict with ``mu_RQ``, ``res_rel`` and the corresponding rate.
        """
        Px = self.propagate_block_projected(x_ritz)
        mu_RQ = jnp.vdot(x_ritz, Px) / jnp.vdot(x_ritz, x_ritz)
        r = Px - mu_RQ * x_ritz
        res_rel = jnp.linalg.norm(r) / jnp.abs(mu_RQ)
        return {
            "mu_RQ": mu_RQ,
            "res_rel": res_rel,
            "rate_RQ": -jnp.log(mu_RQ) / self.T_block,
        }

    def rate_from_mu(self, mu):
        """Target rate ``-log(mu) / T_block`` from an eigenvalue ``mu``."""
        return -jnp.log(mu) / self.T_block