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

import itertools

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow

float_type = gpflow.settings.float_type

class TensorflowTransform(gpflow.transforms.Transform):
    def forward(self, x):
        with tf.Session() as s:
            return s.run(self.forward_tensor(x))

    def backward(self, y):
        with tf.Session() as s:
            return s.run(self.backward_tensor(y))

class CovarianceMatrix(TensorflowTransform):
    def __init__(self, positive_transform=gpflow.transforms.positive):
        self.positive_transform = positive_transform

    def forward_tensor(self, x):
        fwd = tfp.math.fill_triangular(x)
        fwd = tf.matrix_set_diag(fwd, self.positive_transform.forward_tensor(tf.matrix_diag_part(fwd)))
        return fwd

    def backward_tensor(self, y):
        # log transform
        y = tf.matrix_set_diag(
            y, self.positive_transform.backward_tensor(tf.matrix_diag_part(y))
        )

        return tfp.math.fill_triangular_inverse(y)

    def log_jacobian_tensor(self, x):
        raise NotImplementedError()

    def __str__(self):
        return "cov"

class DiagonalMatrix(TensorflowTransform):
    def __init__(self):
        pass

    def forward_tensor(self, x):
        return tf.matrix_diag(x)

    def backward_tensor(self, y):
        return tf.matrix_diag_part(y)

    def log_jacobian_tensor(self, x):
        raise NotImplementedError()

    def __str__(self):
        return "diag"

class Transpose(gpflow.transforms.Transform):
    def __init__(self):
        pass

    def forward(self, x):
        return np.transpose(x)

    def backward(self, y):
        return np.transpose(y)

    def forward_tensor(self, x):
        return tf.transpose(x)

    def backward_tensor(self, y):
        return tf.transpose(y)

    def log_jacobian_tensor(self, x):
        raise NotImplementedError()

    def __str__(self):
        return "transpose"
