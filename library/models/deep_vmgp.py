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
from gpflow.params import Parameter, DataHolder, Minibatch
from gpflow.decors import params_as_tensors, name_scope, autoflow

from .. import transforms
from .. import helper
from ..parametrized_gaussian import MeanFieldGaussian

float_type = gpflow.settings.float_type

# Helper functions
jitter = helper.jitter
cholesky_logdet = helper.cholesky_logdet


class DeepVMGP(gpflow.models.GPModel):
    def __init__(
        self,
        X,
        Y,
        Z_u,
        Z_v,
        w_kerns,
        full_qcov=False,
        diag_qmu=False,
        minibatch_size=None,
        name=None,
    ):
        if minibatch_size is None:
            super().__init__(
                DataHolder(X),
                DataHolder(Y),
                None,
                gpflow.likelihoods.Gaussian(),
                None,
                name=name,
            )
        else:
            super().__init__(
                Minibatch(X, batch_size=minibatch_size, seed=0),
                Minibatch(Y, batch_size=minibatch_size, seed=0),
                None,
                gpflow.likelihoods.Gaussian(),
                None,
                name=name,
            )
        self.full_n = X.shape[0]

        if type(Z_u) is not tuple:
            self.Z_u = gpflow.features.inducingpoint_wrapper(None, Z_u)
            m_u, Q = self.Z_u.Z.shape
        else:
            m_u, Q = Z_u
        if type(Z_v) is not tuple:
            self.Z_v = gpflow.features.inducingpoint_wrapper(None, Z_v)
            m_v, D = self.Z_v.Z.shape
        else:
            m_v, D = Z_v

        n = X.shape[0]
        Dy = Y.shape[1]
        assert D == X.shape[1]
        assert m_v <= n and m_u <= n

        self.is_minibatched = minibatch_size is not None

        self.s_kern = gpflow.kernels.RBF(Q, ARD=False)
        self.s_kern.variance.trainable = False
        self.s_kern.lengthscales.trainable = False

        # Q kernels
        self.w_kerns = gpflow.params.ParamList(w_kerns, name="w_kerns")

        # Parameters
        self.likelihood.variance = 0.01 * np.var(Y)
        self.sigma_f = Parameter(
            np.std(Y), dtype=float_type, transform=gpflow.transforms.positive
        )

        ## q(V) distribution
        # mean: Q * D * m_v
        # cov:  Q * m_v * m_v
        self.qV = MeanFieldGaussian(
            m_v,
            (Q, D),
            # cov(V[:,d1,:]) == cov(V[:,d2,:]) for every d1 and d2
            shared_cov={1},
            full_cov=full_qcov,
            diagonal=diag_qmu,
        )

        if diag_qmu:
            # Q * D * m_v
            self.qV.mean = np.repeat(np.eye(D)[:, :, None], m_v, axis=2)
        else:
            pca_components = PCA(Q).fit(X).components_.astype(float_type)
            # X_proj = X @ pca_components.T
            # input_scales = np.sqrt((2 / np.ptp(X, axis=0) ** 2).reshape(1, -1))
            # Q * D * m_v
            self.qV.mean = np.repeat((pca_components)[:, :, None], m_v, axis=2)

        cov_factor = 1 / D + (0.001 / D)
        batch_eye = helper.batched_identity(m_v, (Q,))
        # Q * m_v * m_v
        self.qV.chol_cov = cov_factor * batch_eye

        ## q(U) distribution
        if self.is_minibatched:
            # mean: Dy * m_v
            # cov:  1 * m_v * m_v
            self.qu = MeanFieldGaussian(
                m_u,
                (Dy,),
                full_cov=True,
            )
        else:
            self.qu = None

        if type(Z_u) is tuple:
            self.Z_u = gpflow.features.InducingPoints(
                helper.initial_inducing_points((X @ self.qV.mean.value[:, :, 0].T), m_u)
            )
        if type(Z_v) is tuple:
            self.Z_v = gpflow.features.InducingPoints(
                helper.initial_inducing_points(X, m_v)
            )

    # m_u * m_u
    @params_as_tensors
    def compute_Ku(self):
        return tf.identity(
            gpflow.features.Kuu(self.Z_u, self.s_kern, jitter=0), name="Ku"
        )

    @params_as_tensors
    def compute_Ku_star(self, Z_star):
        return gpflow.features.Kuf(self.Z_u, self.s_kern, Z_star)

    # Q * m_v * m_v
    @params_as_tensors
    def compute_Kv(self):
        return tf.stack(
            [gpflow.features.Kuu(self.Z_v, k, jitter=0) for k in self.w_kerns]
        )

    # Q * n * n
    @params_as_tensors
    def compute_Kw(self, X=None, diag=False):
        if X is None:
            X = self.X
        if not diag:
            return tf.stack([k.K(X) for k in self.w_kerns], name="Kw")
        else:
            return tf.stack([k.Kdiag(X) for k in self.w_kerns], name="Kw")

    # Q * m_v * n
    @params_as_tensors
    def compute_Kvw(self, X=None):
        if X is None:
            X = self.X
        return tf.stack(
            [gpflow.features.Kuf(self.Z_v, k, X) for k in self.w_kerns], name="Kvw"
        )

    # q(u) ~ Normal with shape (m_u,Dy)
    @params_as_tensors
    def compute_optimal_qu(self):
        m_u = tf.shape(self.Z_u.Z)[0]
        sigma2 = tf.identity(self.likelihood.variance, name="sigma2")
        Psi1 = self.compute_Psi1()
        Psi2 = self.compute_Psi2()

        Ku = self.compute_Ku()
        Ku_Psi2 = sigma2 * Ku + Psi2
        chol_Ku_Psi2 = tf.cholesky(Ku_Psi2 + jitter(m_u), name="chol_Ku_Psi2")

        # m_u * Dy
        Psi1_Y = tf.matmul(Psi1, self.Y, transpose_a=True)

        mean = Ku @ tf.cholesky_solve(chol_Ku_Psi2, Psi1_Y)
        cov = sigma2 * Ku @ tf.cholesky_solve(chol_Ku_Psi2, Ku)

        return mean, cov

    # q(u) ~ Normal with shape (m_u,Dy)
    @params_as_tensors
    def compute_qu(self):
        if not self.is_minibatched:
            return self.compute_optimal_qu()
        else:
            return tf.transpose(self.qu.mean), self.qu.cov[0,:,:]

    # q(s) ~ Normal with shape (n,Dy)
    @params_as_tensors
    def compute_qs(self, Z=None):
        if Z is None:
            Z = self.Z_u.Z
        m_u = tf.shape(self.Z_u.Z)[0]
        Q = tf.shape(self.Z_u.Z)[1]

        # m_u * Dy, m_u * m_u
        qu_mean, qu_cov = self.compute_qu()

        Ku = self.compute_Ku()
        chol_Ku = tf.cholesky(Ku + jitter(m_u), name="chol_Ku")

        # m_u * n
        Ku_star = gpflow.features.Kuf(self.Z_u, self.s_kern, Z)

        # m_u * n
        alpha = tf.cholesky_solve(chol_Ku, Ku_star)

        # n * Dy
        mean = tf.matmul(alpha, qu_mean, transpose_a=True)

        Kstar = self.s_kern.K(Z)
        cov = tf.matmul(alpha, (Ku - qu_cov) @ alpha, transpose_a=True)
        cov = Kstar - cov

        return mean, cov

    # q(f|W) ~ Normal with shape (n,Dy)
    @params_as_tensors
    def compute_qf(self, W, X=None, full_cov=True):
        # n * D
        if X is None:
            X = self.X

        # n * Q * D
        # W

        m_u = tf.shape(self.Z_u.Z)[0]

        # n * Q
        Wx = tf.matmul(
            # n * 1 * D
            X[:, None, :],
            # n * Q * D
            W,
            transpose_b=True,
        )[:, 0, :]

        # m: m_u * Dy
        # S: m_u * m_u
        m, S = self.compute_qu()

        # m_u * m_u
        Ku = self.compute_Ku()
        chol_Ku = tf.cholesky(Ku + jitter(m_u), name="chol_Kv")

        # m_u * n
        Kuf = self.sigma_f * self.compute_Ku_star(Wx)

        # m_u * n
        alpha = tf.cholesky_solve(chol_Ku, Kuf)

        # n * Dy
        mean = tf.matmul(alpha, m, transpose_a=True)

        if full_cov:
            Kf = tf.pow(self.sigma_f, 2) * self.s_kern.K(Wx)
            # n * n
            cov = Kf - tf.matmul(alpha, (Ku - S) @ alpha, transpose_a=True)
        else:
            # n
            Kf = tf.pow(self.sigma_f, 2) * self.s_kern.Kdiag(Wx)
            # n * m_u * 1
            alpha = tf.transpose(alpha[:, :, None], [1, 0, 2])
            # n * 1 * 1
            cov = tf.matmul(alpha, (Ku - S) @ alpha, transpose_a=True)
            # n
            cov = Kf - cov[:, 0, 0]

        return mean, cov

    # q(W) ~ Normal with shape (Q,D,n,1)
    @params_as_tensors
    def compute_qW(self, X=None, full_cov=True):
        if X is None:
            X = self.X
        m_v = tf.shape(self.Z_v.Z)[0]
        D = tf.shape(X)[1]
        Q = tf.shape(self.Z_u.Z)[1]

        Kv = self.compute_Kv()
        Kvw = self.compute_Kvw(X)

        chol_Kv = tf.cholesky(Kv + jitter(m_v), name="chol_Kv")

        # Q * m_v * m_v
        q_cov = self.qV.cov

        # Q * m_v * n
        alpha = tf.cholesky_solve(chol_Kv, Kvw)

        # Q * D * n * 1
        a_tilde = tf.matmul(
            # Q * 1 * m_v * n
            alpha[:, None, :, :],
            # Q * D * m_v * 1
            self.qV.mean[:, :, :, None],
            transpose_a=True,
            name="a_tilde",
        )

        if full_cov:
            # Q * n * n
            Kw = self.compute_Kw(X, diag=False)

            # Q * n * n
            B_tilde = Kw - tf.matmul(alpha, (Kv - q_cov) @ alpha, transpose_a=True)
            # Q * 1 * n * n
            B_tilde = B_tilde[:, None]
        else:
            # Q * n
            Kw = self.compute_Kw(X, diag=True)

            # Q * n * m_v * 1
            alpha = tf.transpose(alpha[:, :, :, None], [0, 2, 1, 3])
            # Q * n * 1 * 1
            B_tilde = tf.matmul(alpha, (Kv - q_cov)[:, None] @ alpha, transpose_a=True)

            # Q * 1 * n
            B_tilde = (Kw - B_tilde[:, :, 0, 0])[:, None, :]

        return a_tilde, B_tilde

    # 1
    @params_as_tensors
    def compute_psi0(self, X=None):
        if X is None:
            X = self.X
        n = tf.shape(X)[0]
        return tf.multiply(tf.cast(n, float_type), tf.pow(self.sigma_f, 2), name="psi0")

    # n * m
    @params_as_tensors
    def compute_Psi1(self, X=None):
        if X is None:
            X = self.X

        # ã  Q * D * n * 1
        # B~ Q * 1 * n
        a_tilde, B_tilde = self.compute_qW(X, full_cov=False)

        # Q * n
        X_a_tilde = tf.matmul(
            # 1 * n * 1 * D
            X[None, :, None, :],
            # Q * n * D * 1
            tf.transpose(a_tilde, [0, 2, 1, 3]),
        )[:, :, 0, 0]

        # Q * n * m_u
        top = tf.subtract(
            # Q * n * 1
            X_a_tilde[:, :, None],
            # Q * 1 * m_u
            tf.transpose(self.Z_u.Z)[:, None, :],
        )
        top = tf.pow(top, 2)

        # Q * n * 1
        bot = (
            # Q * n
            B_tilde[:, 0, :]
            # 1 * n
            * tf.reduce_sum(tf.pow(X, 2), axis=1)[None, :]
            + 1
        )[:, :, None]

        # Q * n * m_u
        Psi1 = tf.exp(-0.5 * top / bot) / tf.sqrt(bot)
        return tf.multiply(self.sigma_f, tf.reduce_prod(Psi1, axis=0), name="Psi1")

    # n * m * m
    @params_as_tensors
    def compute_batched_Psi2(self, X=None):
        if X is None:
            X = self.X
        n = tf.shape(X)[0]
        D = tf.shape(X)[1]
        m_u = tf.shape(self.Z_u.Z)[0]
        Q = tf.shape(self.Z_u.Z)[1]

        # ã  Q * D * n * 1
        # B~ Q * 1 * n
        a_tilde, B_tilde = self.compute_qW(X, full_cov=False)

        # Q * n * 1 * 1
        a_tilde_X = tf.matmul(
            # 1 * n * 1 * D
            X[None, :, None, :],
            # Q * n * D * 1
            tf.transpose(a_tilde, [0, 2, 1, 3]),
        )

        # Q * 1 * m_u * m_u
        Z_bar = tf.transpose(
            0.5 * (self.Z_u.Z[:, None, :] + self.Z_u.Z[None, :, :]), [2, 0, 1]
        )[:, None, :, :]

        # Q * D * n * m_u * m_u
        top = tf.subtract(
            # Q * n * 1 * 1
            a_tilde_X,
            # Q * 1 * m_u * m_u
            Z_bar,
        )
        top = tf.pow(top, 2)

        # Q * n
        bot = 2 * B_tilde[:, 0, :] * tf.reduce_sum(X * X, axis=1)[None, :] + 1

        # Q * n * 1 * 1
        bot = bot[:, :, None, None]

        # Q * D * n * m_u * m_u
        Psi2 = tf.exp(-top / bot) / tf.sqrt(bot)

        # n * m_u * m_u
        Psi2 = tf.reduce_prod(Psi2, axis=0)

        # m_u * m_u * Q
        Z_diff = tf.pow(self.Z_u.Z[:, None, :] - self.Z_u.Z[None, :, :], 2)

        # 1 * m_u * m_u
        Z_diff = tf.exp(-tf.reduce_sum(Z_diff, axis=2) / 4)[None, :, :]

        return tf.multiply(tf.pow(self.sigma_f, 2), Z_diff * Psi2, name="Psi2")

    # m * m
    @params_as_tensors
    def compute_Psi2(self, X=None):
        return tf.reduce_sum(self.compute_batched_Psi2(X), axis=0, name="Psi2")

    @params_as_tensors
    def compute_KL_qV(self):
        m_v = tf.shape(self.Z_v.Z)[0]
        D = tf.shape(self.X)[1]
        Q = tf.shape(self.Z_u.Z)[1]

        # Q * m_v * m_v
        Kv = self.compute_Kv()
        chol_Kv = tf.cholesky(Kv + jitter(m_v), name="chol_Kv")

        # Q * m_v * m_v
        q_cov = self.qV.cov

        # Q * m_v * m_v
        Kvi_q_cov = tf.cholesky_solve(chol_Kv, q_cov)

        KL = tf.cast(D, float_type) * tf.reduce_sum(
            cholesky_logdet(chol_Kv)
            - cholesky_logdet(self.qV.chol_cov)
            + tf.trace(Kvi_q_cov)
        )

        # Q * D * m_v * 1
        batched_q_mu = self.qV.mean[:, :, :, None]

        # Q * D * m_v * m_v
        batched_chol_Kv = tf.tile(chol_Kv[:, None, :, :], [1, D, 1, 1])

        # Q * D * m_v * 1
        KvMu = tf.cholesky_solve(batched_chol_Kv, batched_q_mu, name="KvMu")
        # Q * D
        MuKvMu = tf.matmul(batched_q_mu, KvMu, transpose_a=True)[:, :, 0, 0]

        KL += tf.reduce_sum(MuKvMu, axis=[0, 1])

        KL += -tf.cast(m_v * Q * D, float_type)
        return 0.5 * KL

    @params_as_tensors
    def compute_KL_qu(self):
        m_u = tf.shape(self.Z_u.Z)[0]
        Dy = tf.cast(tf.shape(self.Y)[1], float_type)

        # m_u * m_u
        Ku = self.compute_Ku()
        chol_Ku = tf.cholesky(Ku + jitter(m_u), name="chol_Ku")

        # m: m_u * Dy
        # S: m_u * m_u
        m, S = self.compute_qu()

        if self.qu is not None:
            chol_S = self.qu.chol_cov
        else:
            chol_S = tf.cholesky(S + jitter(m_u), name="chol_S")

        KuiS = tf.cholesky_solve(chol_Ku, S)

        # m_u * Dy
        Kum = tf.cholesky_solve(chol_Ku, m)
        # ()
        mahal = tf.reduce_sum(m * Kum)

        KL = (
            mahal
            + Dy * cholesky_logdet(chol_Ku)
            - Dy * cholesky_logdet(chol_S)
            + Dy * tf.trace(KuiS)
            - Dy * tf.cast(m_u, float_type)
        )
        return 0.5 * KL

    @name_scope("likelihood")
    @params_as_tensors
    def _build_likelihood(self):
        if self.is_minibatched:
            return self._build_minibatch_likelihood()
        else:
            return self._build_collapsed_likelihood()

    @name_scope("likelihood")
    @params_as_tensors
    def _build_minibatch_likelihood(self):
        n = tf.shape(self.X)[0]
        m_v = tf.shape(self.Z_v.Z)[0]
        D = tf.shape(self.X)[1]
        Dy = tf.cast(tf.shape(self.Y)[1], float_type)
        m_u = tf.shape(self.Z_u.Z)[0]
        Q = tf.shape(self.Z_u.Z)[1]

        sigma2 = tf.identity(self.likelihood.variance, name="sigma2")

        # m_u * m_u
        Ku = self.compute_Ku()
        chol_Ku = tf.cholesky(Ku + jitter(m_u), name="chol_Ku")

        # (,)
        YY = tf.reduce_sum(self.Y ** 2)

        # (,)
        psi0 = self.compute_psi0()

        # n * m_u
        Psi1 = self.compute_Psi1()
        # m_u * Dy
        Psi1_Y = tf.matmul(Psi1, self.Y, transpose_a=True)

        # m_u * m_u
        Psi2 = self.compute_Psi2()
        # m_u * m_u
        KuiPsi2 = tf.cholesky_solve(chol_Ku, Psi2)

        # m: m_u * Dy
        # S: m_u * m_u
        m, S = self.compute_qu()

        Kuim = tf.cholesky_solve(chol_Ku, m)

        cross_maha = tf.reduce_sum(Psi1_Y * Kuim)

        maha = tf.reduce_sum((Psi1 @ Kuim) ** 2)

        KuiS = tf.cholesky_solve(chol_Ku, S)

        ## p(y|f)
        F1 = (
            -YY / sigma2
            + 2 * cross_maha / sigma2
            - maha / sigma2
            - Dy * psi0 / sigma2
            - Dy * tf.trace(KuiPsi2) / sigma2
            - Dy * tf.trace(KuiS @ KuiPsi2) / sigma2
            - Dy * tf.cast(n, float_type) * tf.log(2 * tf.cast(np.pi, float_type))
            - Dy * tf.cast(n, float_type) * tf.log(sigma2)
        )

        scale = tf.cast(self.full_n/n, float_type)

        KL_V = self.compute_KL_qV()
        KL_U = self.compute_KL_qu()

        return tf.squeeze((0.5 * F1) - KL_U - KL_V)

    @name_scope("likelihood")
    @params_as_tensors
    def _build_collapsed_likelihood(self):
        n = tf.shape(self.X)[0]
        m_v = tf.shape(self.Z_v.Z)[0]
        D = tf.shape(self.X)[1]
        Dy = tf.cast(tf.shape(self.Y)[1], float_type)
        m_u = tf.shape(self.Z_u.Z)[0]
        Q = tf.shape(self.Z_u.Z)[1]

        sigma2 = tf.identity(self.likelihood.variance, name="sigma2")

        # m_u * m_u
        Ku = self.compute_Ku()
        chol_Ku = tf.cholesky(Ku + jitter(m_u), name="chol_Ku")

        # (,)
        YY = tf.reduce_sum(self.Y ** 2)

        # (,)
        psi0 = self.compute_psi0()

        # m_u * m_u
        Psi2 = self.compute_Psi2()
        # m_u * m_u
        KuiPsi2 = tf.cholesky_solve(chol_Ku, Psi2)

        ## p(y|f) - <U>_q(u)
        F1 = (
            -YY / sigma2
            - Dy * psi0 / sigma2
            + Dy * tf.trace(KuiPsi2) / sigma2
            - Dy * tf.cast(n, float_type) * tf.log(2 * tf.cast(np.pi, float_type))
            - Dy * tf.cast(n - m_u, float_type) * tf.log(sigma2)
        )

        # n * m_u
        Psi1 = self.compute_Psi1()
        # m_u * Dy
        Psi1_Y = tf.matmul(Psi1, self.Y, transpose_a=True)

        # m_u * m_u
        Ku_Psi2 = sigma2 * Ku + Psi2
        # m_u * m_u
        chol_Ku_Psi2 = tf.cholesky(Ku_Psi2 + jitter(m_u), name="chol_Ku_Psi2")

        ## KL(q(u)||p(u)) + <U>_q(u)
        F1 += (
            Dy * cholesky_logdet(chol_Ku)
            - Dy * cholesky_logdet(chol_Ku_Psi2)
            + tf.divide(
                tf.trace(
                    tf.matmul(
                        Psi1_Y,
                        tf.cholesky_solve(chol_Ku_Psi2, Psi1_Y),
                        transpose_a=True,
                    )
                ),
                sigma2,
            )
        )

        KL = self.compute_KL_qV()

        return tf.squeeze((0.5 * F1) - KL)


    # q(F*) ~ Diagonal normal with shape (n*, Dy)
    @params_as_tensors
    def _build_predict(self, X_new, full_cov=False):
        if self.is_minibatched:
            return self._build_minibatch_predict(X_new, full_cov)
        else:
            return self._build_collapsed_predict(X_new, full_cov)

    @params_as_tensors
    def _build_minibatch_predict(self, X_new, full_cov=False):
        assert not full_cov

        m_u = tf.shape(self.Z_u.Z)[0]
        n_new = tf.shape(X_new)[0]

        # m: m_u * Dy
        # S: m_u * m_u
        m,S = self.compute_qu()

        # m_u * m_u
        Ku = self.compute_Ku()
        chol_Ku = tf.cholesky(Ku + jitter(m_u), name='chol_Ku')

        # m_u * Dy
        Kuim = tf.cholesky_solve(chol_Ku, m)

        # n_new * m_u
        Psi1_star = self.compute_Psi1(X_new)

        # n_new * Dy
        mean = Psi1_star @ Kuim

        ## variance

        # Dy * 1 * m_u * 1
        batched_Kuim = tf.transpose(Kuim)[:,None,:,None]

        # n_new * m_u * m_u
        Psi2_star = self.compute_batched_Psi2(X_new)

        # n_new * m_u * m_u
        sq_Psi1_star = tf.matmul(
            Psi1_star[:,None,:], Psi1_star[:,None,:],
            transpose_a=True
        )

        # Dy * n_new
        quadratic_m = tf.matmul(
            batched_Kuim,
            (Psi2_star - sq_Psi1_star) @ batched_Kuim,
            transpose_a=True
        )[:,:,0,0]

        # n_new * m_u * m_u
        batched_chol_Ku = tf.tile(chol_Ku[None, ...], [n_new, 1, 1])
        KuiPsi2_star = tf.cholesky_solve(batched_chol_Ku, Psi2_star)

        KuiS = tf.cholesky_solve(chol_Ku, S)

        # n_new
        trace_terms = (
            tf.trace(KuiPsi2_star)
            - tf.trace(KuiS @ KuiPsi2_star)
        )

        # Dy * n_new
        variance = self.sigma_f**2 + quadratic_m - trace_terms

        return mean, tf.transpose(variance)


    @params_as_tensors
    def _build_collapsed_predict(self, X_new, full_cov=False):
        assert not full_cov

        m_u = tf.shape(self.Z_u.Z)[0]
        n_new = tf.shape(X_new)[0]
        sigma2 = tf.identity(self.likelihood.variance, name="sigma2")

        # m_u * m_u
        Ku = self.compute_Ku()

        # n * m_u
        Psi1 = self.compute_Psi1()

        # m_u * m_u
        Psi2 = self.compute_Psi2()

        Ku_Psi2 = sigma2 * Ku + Psi2
        chol_Ku_Psi2 = tf.cholesky(Ku_Psi2 + jitter(m_u))

        # m_u * Dy
        alpha = tf.cholesky_solve(
            chol_Ku_Psi2, tf.matmul(Psi1, self.Y, transpose_a=True)
        )

        # n_new * m_u
        Psi1_star = self.compute_Psi1(X_new)

        # n_new * Dy
        mean = Psi1_star @ alpha

        chol_Ku = tf.cholesky(Ku + jitter(m_u))

        # n_new * m_u * m_u
        batched_chol_Ku = tf.tile(chol_Ku[None, ...], [n_new, 1, 1])

        # n_new * m_u * m_u
        batched_Psi2_star = self.compute_batched_Psi2(X_new)

        # Dy * 1 * m_u * m_u
        alpha_alpha = tf.matmul(
            tf.transpose(alpha)[:, :, None], tf.transpose(alpha)[:, None, :]
        )[:, None]

        # n_new * m_u * m_u
        batched_chol_Ku_Psi2 = tf.tile(chol_Ku_Psi2[None, ...], [n_new, 1, 1])

        # n_new * Dy
        trace_terms = tf.transpose(
            # 0
            sigma2
            # n_new
            * tf.trace(tf.cholesky_solve(batched_chol_Ku_Psi2, batched_Psi2_star))
            # n_new
            - tf.trace(tf.cholesky_solve(batched_chol_Ku, batched_Psi2_star))
            # Dy * n_new
            + tf.trace(tf.matmul(alpha_alpha, batched_Psi2_star))
        )

        variance = trace_terms - mean ** 2 + self.sigma_f ** 2

        return mean, variance

    @params_as_tensors
    def sample_W(self, X=None, samples=200):
        if X is None:
            X = self.X

        n = tf.shape(X)[0]
        D = tf.shape(X)[1]
        Q = tf.shape(self.Z_u.Z)[1]

        W_eps = tf.random.normal((samples, Q, D, n), dtype=float_type)

        m, S = self.compute_qW(X, full_cov=False)
        m = m[..., 0]

        # samples * Q * D * n
        W = m + W_eps * tf.sqrt(S)

        return W

    @params_as_tensors
    def sample_f(self, X=None, samples=200):
        if X is None:
            X = self.X
        n = tf.shape(X)[0]
        Dy = tf.shape(self.Y)[1]

        # samples * Q * D * n
        W_samples = self.sample_W(X, samples)

        # samples * n * Q * D
        W_samples = tf.transpose(W_samples, [0, 3, 1, 2])

        f_eps = tf.random.normal((samples, n, Dy), dtype=float_type)

        # samples * n * Dy, samples * n
        m, S = tf.map_fn(
            lambda W: self.compute_qf(W, X, full_cov=False),
            W_samples,
            dtype=((float_type, float_type)),
        )

        f = m + f_eps * tf.sqrt(S[..., None])

        return f

    @params_as_tensors
    def sample_y(self, X=None, samples=200):
        if X is None:
            X = self.X
        n = tf.shape(X)[0]
        Dy = tf.shape(self.Y)[1]

        f = self.sample_f(X, samples)
        eps = tf.random.normal((samples, n, Dy), dtype=float_type)

        y = f + eps * tf.sqrt(self.likelihood.variance)

        return y

    @params_as_tensors
    def estimate_ELBO(self, samples=200):
        raise NotImplementedError()
        n = tf.shape(self.X)[0]
        m_v = tf.shape(self.Z_v.Z)[0]
        D = tf.shape(self.X)[1]
        m_u = tf.shape(self.Z_u.Z)[0]
        Q = tf.shape(self.Z_u.Z)[1]

        sigma2 = tf.identity(self.likelihood.variance, name="sigma2")

        # samples * n * 1
        f_samples = self.sample_f(samples=samples)[:, :, None]

        # samples * n * 1
        logpy_samples = tf.divide(
            (
                -tf.pow(self.Y - f_samples, 2) / sigma2
                - tf.log(2 * tf.cast(np.pi, float_type))
                - tf.log(sigma2)
            ),
            2,
        )

        logpy = tf.reduce_sum(tf.reduce_mean(logpy_samples, axis=0), axis=0)[0]

        # qu[0]: m_u * 1
        # qu[1]: m_u * m_u
        qu = self.compute_qu()

        # m_u * m_u
        Ku = self.compute_Ku()
        chol_Ku = tf.cholesky(Ku + jitter(m_u))

        # m_u * m_u
        Ku_qu1 = tf.cholesky_solve(chol_Ku, qu[1])

        # 1 * 1
        maha = tf.matmul(qu[0], tf.cholesky_solve(chol_Ku, qu[0]), transpose_a=True)

        # m_u * m_u
        chol_qu1 = tf.cholesky(qu[1] + jitter(m_u))

        kl_qu = tf.divide(
            (
                tf.trace(Ku_qu1)
                + maha[0, 0]
                - tf.cast(m_u, float_type)
                + cholesky_logdet(chol_Ku)
                - cholesky_logdet(chol_qu1)
            ),
            2,
        )

        kl_qV = self.compute_KL_qV()

        elbo = logpy - kl_qu - kl_qV
        return elbo
