# Copyright 2021 Daniel Augusto Ramos (spectraldani)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from gpflow import Module, Parameter

from library.helper import batched_identity
from library.bijectors import FillDiagonal

float_type = gpflow.config.default_float()

covariance_matrix_bijector = tfp.bijectors.FillScaleTriL(diag_bijector=gpflow.utilities.positive(), diag_shift=None)
diagonal_variance_bijector = tfp.bijectors.Chain([
    FillDiagonal(),
    gpflow.utilities.positive()
], name='FillPositiveDiagonal')


class MeanFieldGaussian(Module):
    def __init__(
            self, n, batch_dims=(), full_cov=True, shared_cov=None, name=None
    ):
        super().__init__(name=name)
        if shared_cov is None:
            shared_cov = set()
        assert shared_cov.intersection(range(len(batch_dims))) == shared_cov, "shared_cov doesn't match batch_dims"

        dims = (*batch_dims, n)
        self.mean = Parameter(np.zeros(dims, dtype=float_type))

        shared_batch = tuple(
            d if i not in shared_cov else 1
            for i, d in enumerate(batch_dims)
        )

        if full_cov:
            self.chol_cov = Parameter(
                batched_identity(n, shared_batch),
                transform=covariance_matrix_bijector,
            )
        else:
            self.chol_cov = Parameter(
                batched_identity(n, shared_batch),
                transform=diagonal_variance_bijector,
            )

    @property
    def cov(self) -> tf.Tensor:
        return tf.matmul(self.chol_cov, self.chol_cov, transpose_b=True, name="cov")
