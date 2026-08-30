from dataclasses import dataclass
from typing import Protocol

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec

from brrrl.types import Metrics, Trajectory, UpdateFn, UpdateRule, UpdateState


class DistributedUpdateRule[S: UpdateState, T: Trajectory](Protocol):
    rule: UpdateRule[S, T]
    # shard_axis_name: str

    def compiled(self, mesh: Mesh, model_sharding: jax.NamedSharding, data_sharding: jax.NamedSharding) -> UpdateFn:
        """jit/shard_map/smap/vmap the `UpdateRule`s `update` function"""
        ...

    def shardings(self) -> tuple[Mesh, jax.NamedSharding, jax.NamedSharding]:
        """Creates a tuple of (mesh, model_sharding, data_sharding)."""
        ...


@dataclass
class DataParallelUpdateRule[S: UpdateState, T: Trajectory]:
    rule: UpdateRule[S, T]
    shard_axis_name: str

    def compiled(self, mesh: Mesh, model_sharding: jax.NamedSharding, data_sharding: jax.NamedSharding) -> UpdateFn:
        def update_and_mean_metrics(state: S, trajectory: T) -> tuple[S, Metrics]:
            state, metrics = self.rule.update(state, trajectory)
            return state, jax.lax.pmean(metrics, self.shard_axis_name)

        return jax.jit(
            jax.shard_map(
                update_and_mean_metrics,
                mesh=mesh,
                in_specs=(model_sharding.spec, data_sharding.spec),
                out_specs=(model_sharding.spec, PartitionSpec()),
            )
        )

    def shardings(self) -> tuple[Mesh, jax.NamedSharding, jax.NamedSharding]:
        # TODO: multi-node training
        learner_devices = jax.local_devices()
        device_mesh = mesh_utils.create_device_mesh((len(learner_devices),), devices=learner_devices)
        mesh = Mesh(device_mesh, axis_names=(self.shard_axis_name,))
        model_spec = PartitionSpec()  # replicate
        data_spec = PartitionSpec(self.shard_axis_name)  # shard
        model_sharding = jax.NamedSharding(mesh, model_spec)
        data_sharding = jax.NamedSharding(mesh, data_spec)

        return mesh, model_sharding, data_sharding
