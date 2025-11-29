"""
Implementation of the Deep Temporal Clustering model
Time Series Clustering layer

@author Florent Forest (FlorentF9)
"""

from tensorflow.keras.layers import Layer, InputSpec
import keras.backend as K


class TSClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, timesteps, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
        dist_metric: distance metric between sequences used in similarity kernel ('eucl', 'cir', 'cor' or 'acf').
    # Input shape
        3D tensor with shape: `(n_samples, timesteps, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, dist_metric='eucl', **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(TSClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.dist_metric = dist_metric
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=3)
        self.clusters = None
        self.built = False

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[2]
        input_steps = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_steps, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_steps, input_dim), initializer='glorot_uniform', name='cluster_centers')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """
        Compute cluster assignment probabilities
        
        # Arguments
            inputs: input tensor with shape (batch_size, timesteps, features)
        # Return
            cluster assignment probabilities
        """
        import tensorflow as tf
        
        # Compute distance between input sequences and cluster centers
        if self.dist_metric == 'eucl':
            # Euclidean distance
            # inputs shape: (batch_size, timesteps, features)
            # self.clusters shape: (n_clusters, timesteps, features)
            # Expand dims to enable broadcasting
            inputs_expanded = tf.expand_dims(inputs, axis=1)  # (batch_size, 1, timesteps, features)
            clusters_expanded = self.clusters  # (n_clusters, timesteps, features)
            
            # Compute squared differences
            diff = inputs_expanded - clusters_expanded  # (batch_size, n_clusters, timesteps, features)
            squared_diff = tf.square(diff)
            
            # Sum over features and timesteps
            sum_features = tf.reduce_sum(squared_diff, axis=3)  # (batch_size, n_clusters, timesteps)
            sqrt_sum = tf.sqrt(sum_features)
            distance = tf.reduce_sum(sqrt_sum, axis=-1)  # (batch_size, n_clusters)
            
        elif self.dist_metric == 'cid':
            # Complexity-Invariant Distance
            import tensorflow as tf
            
            inputs_expanded = tf.expand_dims(inputs, axis=1)
            diff = inputs_expanded - self.clusters
            squared_diff = tf.square(diff)
            sum_features = tf.reduce_sum(squared_diff, axis=3)
            sqrt_sum = tf.sqrt(sum_features)
            distance = tf.reduce_sum(sqrt_sum, axis=-1)
            
            # Compute complexity estimates
            ce_x = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(inputs[:, 1:] - inputs[:, :-1]), axis=-1)), axis=-1, keepdims=True)
            ce_clusters = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(self.clusters[:, 1:] - self.clusters[:, :-1]), axis=-1)), axis=-1)
            
            # Compute CID
            cf = tf.maximum(ce_x, ce_clusters) / tf.minimum(ce_x, ce_clusters)
            distance = distance * cf
            
        elif self.dist_metric == 'cor':
            # Correlation-based distance
            # Normalize inputs
            inputs_mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
            inputs_normalized = inputs - inputs_mean
            
            clusters_mean = tf.reduce_mean(self.clusters, axis=1, keepdims=True)
            clusters_normalized = self.clusters - clusters_mean
            
            # Compute correlation
            inputs_expanded = tf.expand_dims(inputs_normalized, axis=1)
            dot_product = tf.reduce_sum(inputs_expanded * clusters_normalized, axis=[2, 3])
            
            inputs_norm = tf.sqrt(tf.reduce_sum(tf.square(inputs_normalized), axis=[1, 2], keepdims=True))
            clusters_norm = tf.sqrt(tf.reduce_sum(tf.square(clusters_normalized), axis=[1, 2]))
            
            correlation = dot_product / (inputs_norm[:, 0, 0:1] * clusters_norm + 1e-10)
            distance = 1.0 - correlation
            
        elif self.dist_metric == 'acf':
            # Autocorrelation-based distance  
            distance = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.clusters), axis=3)), axis=-1)
        
        # Student's t-distribution kernel
        q = 1.0 / (1.0 + distance**2 / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
        
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters, 'dist_metric': self.dist_metric}
        base_config = super(TSClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
