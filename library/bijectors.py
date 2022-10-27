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

import tensorflow as tf
import tensorflow_probability as tfp

class FillDiagonal(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name='fill_diag'):
        with tf.name_scope(name) as name:
            super(FillDiagonal, self).__init__(
                forward_min_event_ndims=1,
                inverse_min_event_ndims=2,
                is_constant_jacobian=True,
                validate_args=validate_args,
                name=name)

    def _forward(self, x):
        return tf.linalg.diag(x)

    def _inverse(self, y):
        return tf.linalg.diag_part(y)

    @staticmethod
    def _forward_log_det_jacobian(x):
        return tf.zeros([], dtype=x.dtype)

    @staticmethod
    def _inverse_log_det_jacobian(y):
        return tf.zeros([], dtype=y.dtype)

    @property
    def _is_permutation(self):
        return True