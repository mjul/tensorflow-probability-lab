"""
Example of hierarchical linear model using Edward2.
"""
# This is based on the TFP example notebook:
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Linear_Mixed_Effects_Models.ipynb

import tensorflow as tf
from tensorflow_probability import edward2 as ed

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

tf.enable_v2_behavior()

dtype = tf.float64
plt.style.use('ggplot')


def load_insteval():
    """Loads the InstEval data set.

    It contains 73,421 university lecture evaluations by students at ETH
    Zurich with a total of 2,972 students, 2,160 professors and
    lecturers, and several student, lecture, and lecturer attributes.
    Implementation is built from the `observations` Python package.

    Returns:
      Tuple of np.darray `x_train` with 73,421 rows and 7 columns and
      dictionary `metadata` of column headers (feature names).
    """
    url = ('https://raw.github.com/vincentarelbundock/Rdatasets/master/csv/'
           'lme4/InstEval.csv')
    with requests.Session() as s:
        download = s.get(url)
        f = download.content.decode().splitlines()

    iterator = csv.reader(f)
    columns = next(iterator)[1:]
    x_train = np.array([row[1:] for row in iterator], dtype=np.int)
    metadata = {'columns': columns}
    return x_train, metadata


def model(features):
    # Set up fixed effects and other parameters.
    intercept = tf.get_variable("intercept", [])
    service_effects = tf.get_variable("service_effects", [])
    student_stddev_unconstrained = tf.get_variable(
        "student_stddev_pre", [])
    instructor_stddev_unconstrained = tf.get_variable(
        "instructor_stddev_pre", [])  # Set up random effects.
    student_effects = ed.MultivariateNormalDiag(
        loc=tf.zeros(num_students),
        scale_identity_multiplier=tf.exp(
            student_stddev_unconstrained),
        name="student_effects")
    instructor_effects = ed.MultivariateNormalDiag(
        loc=tf.zeros(num_instructors),
        scale_identity_multiplier=tf.exp(
            instructor_stddev_unconstrained),
        name="instructor_effects")  # Set up likelihood given fixed and random effects.
    ratings = ed.Normal(
        loc=(service_effects * features["service"] +
             tf.gather(student_effects, features["students"]) +
             tf.gather(instructor_effects, features["instructors"]) +
             intercept),
        scale=1.,
        name="ratings")
    return ratings


# We load and preprocess the data set. We hold out 20% of the data
# so we can evaluate our fitted model on unseen data points.
# Below we visualize the first few rows.

data, metadata = load_insteval()
data = pd.DataFrame(data, columns=metadata['columns'])
data = data.rename(columns={'s': 'students',
                            'd': 'instructors',
                            'dept': 'departments',
                            'y': 'ratings'})
data['students'] -= 1  # start index by 0
# Remap categories to start from 0 and end at max(category).
data['instructors'] = data['instructors'].astype('category').cat.codes
data['departments'] = data['departments'].astype('category').cat.codes

train = data.sample(frac=0.8)
test = data.drop(train.index)

train.head()


def get_value(dataframe, key, dtype):
    return dataframe[key].values.astype(dtype)


features_train = {
    k: get_value(train, key=k, dtype=np.int32)
    for k in ['students', 'instructors', 'departments', 'service']}

labels_train = get_value(train, key='ratings', dtype=np.float32)

features_test = {k: get_value(test, key=k, dtype=np.int32)
                 for k in ['students', 'instructors', 'departments', 'service']}
labels_test = get_value(test, key='ratings', dtype=np.float32)

num_students = max(features_train['students']) + 1
num_instructors = max(features_train['instructors']) + 1
num_departments = max(features_train['departments']) + 1
num_observations = train.shape[0]

print("Number of students:", num_students)
print("Number of instructors:", num_instructors)
print("Number of departments:", num_departments)
print("Number of observations:", num_observations)


class LinearMixedEffectModel(tf.Module):
    def __init__(self):
        # Set up fixed effects and other parameters.
        # These are free parameters to be optimized in E-steps
        self._intercept = tf.Variable(0., name="intercept")  # alpha in eq
        self._effect_service = tf.Variable(0., name="effect_service")  # beta in eq
        self._stddev_students = tfp.util.TransformedVariable(
            1., bijector=tfb.Exp(), name="stddev_students")  # sigma in eq
        self._stddev_instructors = tfp.util.TransformedVariable(
            1., bijector=tfb.Exp(), name="stddev_instructors")  # sigma in eq
        self._stddev_departments = tfp.util.TransformedVariable(
            1., bijector=tfb.Exp(), name="stddev_departments")  # sigma in eq

    def __call__(self, features):
        model = tfd.JointDistributionSequential([
            # Set up random effects.
            tfd.MultivariateNormalDiag(
                loc=tf.zeros(num_students),
                scale_identity_multiplier=self._stddev_students),
            tfd.MultivariateNormalDiag(
                loc=tf.zeros(num_instructors),
                scale_identity_multiplier=self._stddev_instructors),
            tfd.MultivariateNormalDiag(
                loc=tf.zeros(num_departments),
                scale_identity_multiplier=self._stddev_departments),
            # This is the likelihood for the observed.
            lambda effect_departments, effect_instructors, effect_students: tfd.Independent(
                tfd.Normal(
                    loc=(self._effect_service * features["service"] +
                         tf.gather(effect_students, features["students"], axis=-1) +
                         tf.gather(effect_instructors, features["instructors"], axis=-1) +
                         tf.gather(effect_departments, features["departments"], axis=-1) +
                         self._intercept),
                    scale=1.),
                reinterpreted_batch_ndims=1)
        ])

        # To enable tracking of the trainable variables via the created distribution,
        # we attach a reference to `self`. Since all TFP objects sub-class
        # `tf.Module`, this means that the following is possible:
        # LinearMixedEffectModel()(features_train).trainable_variables
        # ==> tuple of all tf.Variables created by LinearMixedEffectModel.
        model._to_track = self
        return model


lmm_jointdist = LinearMixedEffectModel()
# Conditioned on feature/predictors from the training data
lmm_train = lmm_jointdist(features_train)

print("Trainable variables", lmm_train.trainable_variables)

lmm_train.resolve_graph()

target_log_prob_fn = lambda *x: lmm_train.log_prob(x + (labels_train,))
trainable_variables = lmm_train.trainable_variables
current_state = lmm_train.sample()[:-1]

# For debugging
target_log_prob_fn(*current_state)

# Set up E-step (MCMC).
hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    step_size=0.015,
    num_leapfrog_steps=3)
kernel_results = hmc.bootstrap_results(current_state)


@tf.function(autograph=False, experimental_compile=False)
def one_e_step(current_state, kernel_results):
    next_state, next_kernel_results = hmc.one_step(
        current_state=current_state,
        previous_kernel_results=kernel_results)
    return next_state, next_kernel_results


optimizer = tf.optimizers.Adam(learning_rate=.01)


# Set up M-step (gradient descent).
@tf.function(autograph=False, experimental_compile=False)
def one_m_step(current_state):
    with tf.GradientTape() as tape:
        loss = -target_log_prob_fn(*current_state)
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    return loss


num_warmup_iters = 1000
num_iters = 1500
num_accepted = 0
effect_students_samples = np.zeros([num_iters, num_students])
effect_instructors_samples = np.zeros([num_iters, num_instructors])
effect_departments_samples = np.zeros([num_iters, num_departments])
loss_history = np.zeros([num_iters])

# Run warm-up stage.
for t in range(num_warmup_iters):
    current_state, kernel_results = one_e_step(current_state, kernel_results)
    num_accepted += kernel_results.is_accepted.numpy()
    if t % 500 == 0 or t == num_warmup_iters - 1:
        print("Warm-Up Iteration: {:>3} Acceptance Rate: {:.3f}".format(
            t, num_accepted / (t + 1)))

num_accepted = 0  # reset acceptance rate counter

# Run training.
for t in range(num_iters):
    # run 5 MCMC iterations before every joint EM update
    for _ in range(5):
        current_state, kernel_results = one_e_step(current_state, kernel_results)
    loss = one_m_step(current_state)
    effect_students_samples[t, :] = current_state[0].numpy()
    effect_instructors_samples[t, :] = current_state[1].numpy()
    effect_departments_samples[t, :] = current_state[2].numpy()
    num_accepted += kernel_results.is_accepted.numpy()
    loss_history[t] = loss.numpy()
    if t % 500 == 0 or t == num_iters - 1:
        print("Iteration: {:>4} Acceptance Rate: {:.3f} Loss: {:.3f}".format(
            t, num_accepted / (t + 1), loss_history[t]))


@tf.function(autograph=False, experimental_compile=False)
def run_k_e_steps(k, current_state, kernel_results):
    _, next_state, next_kernel_results = tf.while_loop(
        cond=lambda i, state, pkr: i < k,
        body=lambda i, state, pkr: (i + 1, *one_e_step(state, pkr)),
        loop_vars=(tf.constant(0), current_state, kernel_results)
    )
    return next_state, next_kernel_results


plt.plot(loss_history)
plt.ylabel(r'Loss $-\log$ $p(y\mid\mathbf{x})$')
plt.xlabel('Iteration')
plt.show()

for i in range(7):
    plt.plot(effect_instructors_samples[:, i])

plt.legend([i for i in range(7)], loc='lower right')
plt.ylabel('Instructor Effects')
plt.xlabel('Iteration')
plt.show()

# Criticism

lmm_test = lmm_jointdist(features_test)

[
    effect_students_mean,
    effect_instructors_mean,
    effect_departments_mean,
] = [
    np.mean(x, axis=0).astype(np.float32) for x in [
        effect_students_samples,
        effect_instructors_samples,
        effect_departments_samples
    ]
]

# Get the posterior predictive distribution
(*posterior_conditionals, ratings_posterior), _ = lmm_test.sample_distributions(
    value=(
        effect_students_mean,
        effect_instructors_mean,
        effect_departments_mean,
    ))

ratings_prediction = ratings_posterior.mean()

plt.title("Residuals for Predicted Ratings on Test Set")
plt.xlim(-4, 4)
plt.ylim(0, 800)
plt.hist(ratings_prediction - labels_test, 75)
plt.show()

plt.title("Histogram of Student Effects")
plt.hist(effect_students_mean, 75)
plt.show()

plt.title("Histogram of Instructor Effects")
plt.hist(effect_instructors_mean, 75)
plt.show()

plt.title("Histogram of Department Effects")
plt.hist(effect_departments_mean, 75)
plt.show()
