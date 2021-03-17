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

import sys

import gpflow
import tensorflow as tf
import numpy as np

from sklearn.decomposition import PCA
from gpflow.params import Parameter
from gpflow.decors import params_as_tensors, name_scope

from .. import helper

float_type = gpflow.settings.float_type

# Helper functions
jitter = helper.jitter
cholesky_logdet = helper.cholesky_logdet


class VMGP(gpflow.models.GPModel):
    def __init__(self, X, Y, Z, name=None):
        super().__init__(X, Y, None, gpflow.likelihoods.Gaussian(), None, name=name)

        if type(Z) is not tuple:
            self.feature = gpflow.features.inducingpoint_wrapper(None, Z)
            K = self.feature.Z.shape[1]
        else:
            M, K = Z

        self.s_kern = gpflow.kernels.RBF(K)
        self.s_kern.variance.trainable = False
        self.s_kern.lengthscales.trainable = False
        D = self.X.shape[1]
        # Parameters
        pca_components = PCA(K).fit(X).components_.astype(float_type)
        X_proj = X @ pca_components.T
        input_scales = (10 / np.ptp(X_proj, axis=0) ** 2).reshape(-1, 1)

        self.likelihood.variance = 0.01 * np.var(Y)
        self.sigma_f = Parameter(
            np.std(Y), dtype=float_type, transform=gpflow.transforms.positive
        )
        self.q_mu = Parameter((pca_components * input_scales), dtype=float_type)
        self.q_cov = Parameter(
            (1 / D + (0.001 / D) * np.random.randn(K, D)),
            dtype=float_type,
            transform=gpflow.transforms.positive,
        )

        if type(Z) is tuple:
            self.feature = gpflow.features.InducingPoints(
                helper.initial_inducing_points(X @ self.q_mu.value.T, M)
            )

    # 1
    @params_as_tensors
    def compute_psi0(self):
        n = tf.shape(self.X)[0]
        return tf.multiply(tf.cast(n, float_type), self.sigma_f ** 2, name="psi0")

    # n * m
    @params_as_tensors
    def compute_Psi1(self, X=None):
        if X is None:
            X = self.X

        # K * n
        q_mu_X = tf.tensordot(self.q_mu, X, [[1], [1]], name="q_mu_x")

        with tf.name_scope("Psi1"):
            # K * n * m
            top = tf.subtract(
                q_mu_X[:, :, None], tf.transpose(self.feature.Z, [1, 0])[:, None, :]
            )
            top = tf.pow(top, 2, name="top")

            # K * n
            bot = 1 + tf.reduce_sum(self.q_cov[:, None, :] * tf.pow(X[None], 2), axis=2)
            # K * n * 1
            bot = tf.identity(bot[:, :, None], name="bot")

            # K * n * m
            Psi1 = tf.exp(-0.5 * top / bot) / tf.sqrt(bot)

        return tf.multiply(self.sigma_f, tf.reduce_prod(Psi1, axis=0), name="Psi1")

    # n * m * m
    @params_as_tensors
    def compute_batched_Psi2(self, X=None):
        if X is None:
            X = self.X

        # K * n
        q_mu_X = tf.tensordot(self.q_mu, X, [[1], [1]], name="q_mu_x")

        with tf.name_scope("Psi2"):
            # m * m * K
            Z_diff = tf.subtract(
                self.feature.Z[:, None, :], self.feature.Z[None, :, :], name="Z_diff"
            )

            # m * m * K
            Z_bar = 0.5 * (self.feature.Z[:, None, :] + self.feature.Z[None, :, :])

            # K * n * m * m
            top = tf.subtract(
                q_mu_X[:, :, None, None], tf.transpose(Z_bar, [2, 0, 1])[:, None, :, :]
            )
            top = tf.pow(top, 2)

            # K * n
            bot = (
                2 * tf.reduce_sum(self.q_cov[:, None, :] * tf.pow(X[None], 2), axis=2)
                + 1
            )

            # K * n * 1 * 1
            bot = tf.identity(bot[:, :, None, None], name="bot")

            # K * n * m * m
            right = tf.exp(-top / bot) / tf.sqrt(bot)
            # n * m * m
            right = tf.reduce_prod(right, axis=0, name="right")

            # m * m
            left = tf.multiply(
                tf.pow(self.sigma_f, 2),
                tf.exp(-0.25 * tf.reduce_sum(tf.pow(Z_diff, 2), axis=2)),
                name="left",
            )
        return tf.multiply(left, right, name="BatchedPsi2")

    @params_as_tensors
    def compute_Psi2(self, X=None):
        return tf.reduce_sum(self.compute_batched_Psi2(X), axis=0, name="Psi2")

    @name_scope("likelihood")
    @params_as_tensors
    def _build_likelihood(self):
        n = tf.shape(self.X)[0]
        D = tf.shape(self.X)[1]
        m = tf.shape(self.feature.Z)[0]

        sigma2 = self.likelihood.variance

        # m * m
        Ku = gpflow.features.Kuu(self.feature, self.s_kern, jitter=0.0)
        chol_Ku = tf.cholesky(Ku + jitter(m), name="chol_Ku")

        F1 = (
            -tf.cast(n, float_type) * tf.log(2 * tf.cast(np.pi, float_type))
            - tf.cast(n - m, float_type) * tf.log(sigma2)
            + cholesky_logdet(chol_Ku)
        )

        # 1 * 1
        YY = tf.matmul(self.Y, self.Y, transpose_a=True)

        # m * m
        Psi2 = self.compute_Psi2()

        # m * m
        Ku_Psi2 = sigma2 * Ku + Psi2
        # m * m
        chol_Ku_Psi2 = tf.cholesky(Ku_Psi2 + jitter(m), name="chol_Ku_Psi2")

        F1 += -cholesky_logdet(chol_Ku_Psi2) - YY / sigma2

        # n * m
        Psi1 = self.compute_Psi1()

        # m * 1
        Psi1_Y = tf.matmul(Psi1, self.Y, transpose_a=True)

        F1 += tf.divide(
            tf.matmul(
                Psi1_Y, tf.cholesky_solve(chol_Ku_Psi2, Psi1_Y), transpose_a=True
            ),
            sigma2,
        )

        # 1
        psi0 = self.compute_psi0()

        # m * m
        KuiPsi2 = tf.cholesky_solve(chol_Ku, Psi2)

        F1 += -psi0 / sigma2 + tf.trace(KuiPsi2) / sigma2

        KL = tf.reduce_sum(tf.log(self.q_cov), axis=1)
        KL -= tf.cast(D, float_type) * tf.log(
            tf.reduce_sum(self.q_cov + tf.pow(self.q_mu, 2), axis=1)
        )
        KL += tf.cast(D, float_type) * tf.log(tf.cast(D, float_type))
        KL = tf.reduce_sum(KL)

        return tf.squeeze(0.5 * (F1 + KL))

    @name_scope("predict")
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        n = tf.shape(self.X)[0]
        nNew = tf.shape(Xnew)[0]
        D = tf.shape(self.X)[1]
        m = tf.shape(self.feature.Z)[0]
        K = tf.shape(self.feature.Z)[1]

        sigma2 = self.likelihood.variance
        Ku = gpflow.features.Kuu(self.feature, self.s_kern, jitter=0)
        chol_Ku = tf.cholesky(Ku + jitter(m), name="chol_Ku")

        Psi1 = self.compute_Psi1()
        Psi2 = self.compute_Psi2()

        Ku_Psi2 = sigma2 * Ku + Psi2
        chol_Ku_Psi2 = tf.cholesky(Ku_Psi2 + jitter(m), name="chol_Ku_Psi2")

        Psi1_Y = tf.matmul(Psi1, self.Y, transpose_a=True)
        alpha = tf.cholesky_solve(chol_Ku_Psi2, Psi1_Y)

        Psi1new = self.compute_Psi1(Xnew)

        mean = Psi1new @ alpha

        Psi2new = self.compute_batched_Psi2(Xnew)
        var = tf.trace(
            sigma2
            * tf.cholesky_solve(tf.tile(chol_Ku_Psi2[None], [nNew, 1, 1]), Psi2new)
            - tf.cholesky_solve(tf.tile(chol_Ku[None], [nNew, 1, 1]), Psi2new)
            + tf.tile(tf.matmul(alpha, alpha, transpose_b=True)[None], [nNew, 1, 1])
            @ Psi2new
        )
        var = tf.reshape(var, (-1, 1)) + self.sigma_f ** 2 + sigma2 - mean ** 2

        return mean, var

    @gpflow.decors.autoflow((float_type, [None, None]))
    def predict_y(self, Xnew):
        pred_y_mean, pred_y_var = self._build_predict(Xnew)
        return pred_y_mean, pred_y_var

    # q(u) ~ Normal with shape (m,1)
    @params_as_tensors
    def compute_qu(self):
        m = tf.shape(self.feature.Z)[0]
        sigma2 = tf.identity(self.likelihood.variance, name="sigma2")
        Psi1 = self.compute_Psi1()
        Psi2 = self.compute_Psi2()

        Ku = gpflow.features.Kuu(self.feature, self.s_kern, jitter=0.0)
        Ku_Psi2 = sigma2 * Ku + Psi2
        chol_Ku_Psi2 = tf.cholesky(Ku_Psi2 + jitter(m), name="chol_Ku_Psi2")
        Psi1_Y = tf.matmul(Psi1, self.Y, transpose_a=True)

        mean = Ku @ tf.cholesky_solve(chol_Ku_Psi2, Psi1_Y)
        cov = sigma2 * Ku @ tf.cholesky_solve(chol_Ku_Psi2, Ku)

        return mean, cov
