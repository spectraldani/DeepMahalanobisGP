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
import numpy as np

from gpflow.params import Parameter, Parameterized
from gpflow.decors import params_as_tensors
from . import transforms
from .helper import batched_identity

float_type = gpflow.settings.float_type


class MeanFieldGaussian(Parameterized):
    def __init__(
        self, n, batch_dims=(), full_cov=True, diagonal=False, shared_cov={}, name=None
    ):
        super().__init__(name=name)
        dims = (*batch_dims, n)

        if len(batch_dims) > 1 and set(shared_cov) != set(range(1, len(batch_dims))):
            raise NotImplementedError("Covariance must be shared in all dimensions")

        if not diagonal:
            self.mean = Parameter(np.zeros(dims, dtype=float_type))

            # TODO -- HACK
            if len(batch_dims) == 1 and set(shared_cov) == {1}:
                batch_dims[0] = 1

            if full_cov:
                self.chol_cov = Parameter(
                    batched_identity(n, (batch_dims[0],)),
                    transform=transforms.CovarianceMatrix(n, batch_dims[0]),
                )
            else:
                self.chol_cov = Parameter(
                    batched_identity(n, (batch_dims[0],)),
                    transform=transforms.DiagonalMatrix()(gpflow.transforms.positive),
                )
        else:
            assert len(batch_dims) >= 2, "Must be matrix or tensor"
            assert np.equal.reduce(batch_dims), "Must have equal dimensions"
            if len(batch_dims) != 2:
                raise NotImplementedError("Not implemented for tensors")
            if not full_cov:
                raise NotImplementedError("Covariance must be full")

            self.mean = Parameter(
                np.zeros(dims, dtype=float_type),
                transform=transforms.Transpose()(transforms.DiagonalMatrix()),
            )
            self.chol_cov = Parameter(
                batched_identity(n, (batch_dims[0],)),
                transform=transforms.CovarianceMatrix(n, batch_dims[0]),
            )

    @property
    @params_as_tensors
    def cov(self):
        return tf.matmul(self.chol_cov, self.chol_cov, transpose_b=True, name="cov")
