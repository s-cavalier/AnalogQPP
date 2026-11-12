import equinox as eq
import jax
import jax.numpy as jnp

from curve import LearnableCurve

class BSplineCurve(LearnableCurve):
    num_control_points: int = eq.field(static=True)
    invertible: bool = eq.field(static=True)
    endpoint: jnp.ndarray
    free_control_points: jnp.ndarray
    initial_velocity_log_scale: jnp.ndarray
    final_velocity_log_scale: jnp.ndarray | None
    knots: jnp.ndarray = eq.field(repr=False)

    def __init__(
        self,
        n_ctrl_pts: int,
        endpoint,
        invertible: bool,
        key: jnp.ndarray | None = None,
        init_scale: float = 0.1,
    ):
        n_ctrl_pts = n_ctrl_pts.__index__()
        if n_ctrl_pts < 6: raise ValueError("n_ctrl_pts must be at least 6 for a quintic B-spline.")

        endpoint = jnp.asarray(endpoint)
        if endpoint.shape != (3,): raise ValueError("endpoint must have shape (3,).")
        if not jnp.issubdtype(endpoint.dtype, jnp.floating):
            if jnp.issubdtype(endpoint.dtype, jnp.complexfloating):
                raise ValueError("endpoint must be real-valued.")
            endpoint = endpoint.astype(jnp.asarray(0.0).dtype)

        self.num_control_points = n_ctrl_pts
        self.invertible = bool(invertible)
        self.endpoint = endpoint
        dtype = endpoint.dtype
        spans = n_ctrl_pts - 5
        self.knots = jnp.concatenate(
            [
                jnp.zeros(6, dtype=dtype),
                jnp.arange(1, spans, dtype=dtype) / spans,
                jnp.ones(6, dtype=dtype),
            ])

        if key is None: key = jax.random.key(0)
        free_key, initial_key, final_key = jax.random.split(key, 3)
        free_count = n_ctrl_pts - (6 if self.invertible else 5)
        scale = jnp.asarray(init_scale, dtype=dtype)
        noise_scale = scale * jnp.maximum(jnp.linalg.norm(endpoint), 2.0)
        self.free_control_points = noise_scale * jax.random.normal(free_key, (free_count, 3), dtype=dtype,)
        self.initial_velocity_log_scale = scale * jax.random.normal(initial_key, (), dtype=dtype)
        self.final_velocity_log_scale = (scale * jax.random.normal(final_key, (), dtype=dtype) if self.invertible else None)
        super().__init__(0.0, 1.0)

    def _points(self, indices: jnp.ndarray) -> jnp.ndarray:
        n = self.num_control_points
        z_hat = jnp.array([0.0, 0.0, 1.0], dtype=self.endpoint.dtype)
        initial_velocity = jnp.exp(self.initial_velocity_log_scale) * z_hat
        first = initial_velocity * self.knots[6] / 5
        second = first + initial_velocity * self.knots[7] / 5

        if self.invertible:
            final_velocity = -jnp.exp(self.final_velocity_log_scale) * z_hat
            penultimate = self.endpoint - final_velocity * (1 - self.knots[n - 1]) / 5
            antepenultimate = penultimate - final_velocity * (1 - self.knots[n - 2]) / 5
        else:
            penultimate = self.free_control_points[-1]
            ratio = (1 - self.knots[n - 2]) / (1 - self.knots[n - 1])
            antepenultimate = penultimate - ratio * (self.endpoint - penultimate)

        if self.free_control_points.shape[0]:
            points = jnp.take(self.free_control_points, indices - 3, axis=0, mode="clip")
        else:
            points = jnp.zeros(indices.shape + (3,), dtype=self.endpoint.dtype)
        for index, point in [
            (0, jnp.zeros(3, dtype=self.endpoint.dtype)),
            (1, first),
            (2, second),
            (n - 3, antepenultimate),
            (n - 2, penultimate),
            (n - 1, self.endpoint),
        ]:
            points = jnp.where((indices == index)[..., None], point, points)
        return points

    def _control_points(self) -> jnp.ndarray: return self._points(jnp.arange(self.num_control_points))

    def trainable_filter_spec(self):
        spec = super().trainable_filter_spec()
        return eq.tree_at(lambda tree: (tree.endpoint, tree.knots), spec, replace=(False, False))

    def _evaluate(self, t: jnp.ndarray, derivative: int) -> jnp.ndarray:
        t = jnp.asarray(t, dtype=self.endpoint.dtype)
        spans = self.num_control_points - 5
        span = jnp.clip(jnp.floor(t * spans).astype(jnp.int32), 0, spans - 1) + 5
        start = span - 5
        points = self._points(start[..., None] + jnp.arange(6))

        for order in range(1, derivative + 1):
            indices = start[..., None] + jnp.arange(6 - order)
            widths = self.knots[indices + 6] - self.knots[indices + order]
            points = (6 - order) * (points[..., 1:, :] - points[..., :-1, :]) / widths[..., None]

        degree = 5 - derivative
        for level in range(1, degree + 1):
            indices = start[..., None] + jnp.arange(level, degree + 1)
            left = self.knots[indices + derivative]
            right = self.knots[indices + 6 - level]
            alpha = (t[..., None] - left) / (right - left)
            points = (1 - alpha[..., None]) * points[..., :-1, :] + alpha[..., None] * points[..., 1:, :]
        return points[..., 0, :]

    @eq.filter_jit
    def position(self, t: jnp.ndarray) -> jnp.ndarray: return self._evaluate(t, 0)

    @eq.filter_jit
    def velocity(self, t: jnp.ndarray) -> jnp.ndarray: return self._evaluate(t, 1)

    @eq.filter_jit
    def acceleration(self, t: jnp.ndarray) -> jnp.ndarray: return self._evaluate(t, 2)

    @eq.filter_jit
    def jerk(self, t: jnp.ndarray) -> jnp.ndarray: return self._evaluate(t, 3)
