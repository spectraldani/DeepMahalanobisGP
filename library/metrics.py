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

import numpy as np

def diagonalize_predict_var(predict_var):
    if predict_var.shape[1] == predict_var.shape[0]:
        return np.diag(predict_var).reshape(-1,1)
    else:
        return predict_var

def root_mean_squared_error(observed_value, predict_mean, predict_var=None):
    se = (observed_value - predict_mean) ** 2
    return np.sqrt(np.mean(se,axis=0))

def negative_log_predictive_density(observed_value, predict_mean, predict_var):
    n = observed_value.shape[0]
    predict_var = diagonalize_predict_var(predict_var)
    inner = np.log(predict_var) + (observed_value - predict_mean) ** 2 / predict_var
    return 0.5 * (np.log(2 * np.pi) + np.mean(inner, axis=0))

def mean_relative_absolute_error(observed_value, predicted_mean, predict_var=None):
    ae = np.abs(observed_value - predicted_mean)
    return np.mean(ae/np.abs(observed_value),axis=0)
