from pykalman import KalmanFilter
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints # Import the sigma points method
from filterpy.monte_carlo import systematic_resample
from numpy.random import uniform, normal
###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def experiment_ekf(df_train, df_test, predict_cols, n_latent_variables = 10):

    # Define the number of latent variables and observation dimensions
     # Corrected the number of latent variables to match the number of observed variables
    n_observation_dimensions = len(predict_cols) # We are observing all columns in predict_cols

    # Define non-linear transition function (placeholder - replace with your actual function)
    # State is based on the latent variables
    def transition_function(state):
        # Example: Simple non-linear transition
        # Next state is a non-linear combination of the current state
        return state + 0.1 * np.sin(state) # Placeholder non-linear transition

    # Define the Jacobian of the transition function (placeholder - replace with your actual Jacobian)
    def transition_covariance(state):
        # Example: Jacobian of the simple non-linear transition
        # Jacobian is a matrix of partial derivatives
        return np.eye(n_latent_variables) + 0.1 * np.diag(np.cos(state).flatten()) # Placeholder Jacobian

    # Define non-linear observation function (placeholder - replace with your actual function)
    # Observation is based on the latent variables
    def observation_function(state):
        # Example: Non-linear observation of the state
        # Assuming the observation is a direct mapping from the latent state for now
        return state[:n_observation_dimensions] # Observe the first n_observation_dimensions states


    # Define the Jacobian of the observation function (placeholder - replace with your actual Jacobian)
    def observation_covariance(state):
        # Example: Jacobian of the non-linear observation function
        # Assuming the observation is a direct mapping from the latent state, the Jacobian is identity
        H_k = np.zeros((n_observation_dimensions, n_latent_variables))
        H_k[:, :n_observation_dimensions] = np.eye(n_observation_dimensions)
        return H_k


    # Train a basic Kalman Filter with n_latent_variables to get the initial state for EKF
    kf_initial = KalmanFilter(transition_matrices=np.eye(n_latent_variables),
                            observation_matrices=np.ones((n_observation_dimensions, n_latent_variables)), # Corrected observation matrix shape
                            initial_state_mean=np.zeros(n_latent_variables),
                            initial_state_covariance=np.eye(n_latent_variables),
                            observation_covariance=np.eye(n_observation_dimensions),
                            transition_covariance=np.eye(n_latent_variables) * 0.01)

    state_means_initial, state_covariances_initial = kf_initial.filter(df_train[predict_cols].values)

    # Initial state mean and covariance from the basic KF training
    initial_state_mean = state_means_initial[-1].reshape(-1, 1)
    initial_state_covariance = state_covariances_initial[-1]


    # Noise covariances (these are typically tuned)
    process_noise_covariance = np.eye(n_latent_variables) * 0.01 # Q
    observation_noise_covariance = np.eye(n_observation_dimensions) # R


    # Predict and Update steps for Extended Kalman Filter
    current_state_mean = initial_state_mean
    current_state_covariance = initial_state_covariance

    predicted_means = []
    predicted_covariances = []

    for i in range(len(df_test)):
        # EKF Prediction Step
        # Predict the next state using the non-linear transition function
        predicted_state_mean = transition_function(current_state_mean)

        # Calculate the Jacobian of the transition function at the current state
        F_k = transition_covariance(current_state_mean)

        # Predict the next state covariance
        predicted_state_covariance = F_k @ current_state_covariance @ F_k.T + process_noise_covariance

        # Get the actual observation from df_test for the current time step
        actual_observation = df_test[predict_cols].iloc[max(i-1,0)].values.reshape(-1, 1)

        # EKF Update Step
        # Predict the observation based on the predicted state using the non-linear observation function
        predicted_observation_mean = observation_function(predicted_state_mean)

        # Calculate the Jacobian of the observation function at the predicted state
        H_k = observation_covariance(predicted_state_mean)

        # Calculate the innovation (measurement residual)
        innovation = actual_observation - predicted_observation_mean

        # Calculate the innovation covariance
        innovation_covariance = H_k @ predicted_state_covariance @ H_k.T + observation_noise_covariance

        # Calculate the Kalman Gain
        kalman_gain = predicted_state_covariance @ H_k.T @ np.linalg.inv(innovation_covariance)

        # Update the state mean and covariance
        current_state_mean = predicted_state_mean + kalman_gain @ innovation
        current_state_covariance = predicted_state_covariance - kalman_gain @ H_k @ predicted_state_covariance

        # Append the predicted observation (based on the updated state)
        predicted_means.append(observation_function(current_state_mean).flatten())
        predicted_covariances.append(current_state_covariance.diagonal())


    # Convert predicted means to a numpy array
    predicted_means = np.array(predicted_means)


    # Create a DataFrame for the predicted values with the same index as df_test
    df_predicted_ekf = pd.DataFrame(predicted_means, index=df_test.index, columns=predict_cols)

    # df_predicted_ekf_hour = df_predicted_ekf.loc[start_time:end_time]

    return df_predicted_ekf

def experiment_ukf(df_train, df_test, predict_cols, n_latent_variables = 10):

    # Define the state transition function (non-linear example)
    def state_transition_ukf(x, dt):
        """
        Non-linear state transition function for UKF.
        x: current state (numpy array)
        dt: time step (scalar)
        """
        # Example: Simple non-linear transition
        # This should be adapted to your specific problem's dynamics
        return x + 0.01 * np.sin(x * dt) # Placeholder non-linear transition

    # Define the observation function (non-linear example)
    def observation_function_ukf(x):
        """
        Non-linear observation function for UKF.
        x: current state (numpy array)
        """
        # Example: Observe a non-linear combination of the state
        # Assuming the observation is a direct mapping from the latent state for now
        return x[:n_observation_dimensions] # Observe the first n_observation_dimensions states


    # Define the number of latent variables and observation dimensions
    n_latent_variables = 10 # Corrected to match the EKF initialization
    n_observation_dimensions = len(predict_cols) # We are observing all columns in predict_cols

    # Initialize UKF
    # You need to define process_noise_covariance (Q) and observation_noise_covariance (R)
    Q = np.eye(n_latent_variables) * 0.01 # Process noise covariance
    R = np.eye(n_observation_dimensions) # Observation noise covariance

    # Train a basic Kalman Filter with n_latent_variables to get the initial state for EKF
    kf_initial = KalmanFilter(transition_matrices=np.eye(n_latent_variables),
                            observation_matrices=np.ones((n_observation_dimensions, n_latent_variables)), # Corrected observation matrix shape
                            initial_state_mean=np.zeros(n_latent_variables),
                            initial_state_covariance=np.eye(n_latent_variables),
                            observation_covariance=np.eye(n_observation_dimensions),
                            transition_covariance=np.eye(n_latent_variables) * 0.01)

    state_means_initial, state_covariances_initial = kf_initial.filter(df_train[predict_cols].values)

    # Initial state mean and covariance from the basic KF training
    initial_state_mean = state_means_initial[-1].reshape(-1, 1)
    initial_state_covariance = state_covariances_initial[-1]

    # Initial state mean and covariance (can be estimated from training data or prior knowledge)
    # Using the last state mean and covariance from the EKF for initialization
    # Need to ensure state_means_initial and state_covariances_initial are available from previous cell execution
    # If not, you might need to re-run the EKF initialization part or load them if saved.
    # For now, assuming they are available in the environment.
    initial_state_mean_ukf = state_means_initial[-1]
    initial_state_covariance_ukf = state_covariances_initial[-1]

    # Define sigma points
    points = MerweScaledSigmaPoints(n_latent_variables, alpha=.1, beta=2., kappa=0.)

    ukf = UKF(dim_x=n_latent_variables, dim_z=n_observation_dimensions, fx=state_transition_ukf, hx=observation_function_ukf, dt=1., points=points) # Pass sigma points to UKF

    # Set initial state mean and covariance
    ukf.x = initial_state_mean_ukf
    ukf.P = initial_state_covariance_ukf

    # Set noise covariances
    ukf.R = R
    ukf.Q = Q


    # Predict and Update steps for UKF
    predicted_means_ukf = []

    for i in range(len(df_test)):
        # Get the actual observation for the current time step
        actual_observation = df_test[predict_cols].iloc[max(i - 1,0)].values

        # UKF Prediction Step
        ukf.predict()

        # UKF Update Step
        ukf.update(actual_observation)

        # Append the predicted observation (based on the updated state)
        predicted_means_ukf.append(ukf.x[:n_observation_dimensions])


    # Convert predicted means to a numpy array
    predicted_means_ukf = np.array(predicted_means_ukf)

    # Create a DataFrame for the predicted values with the same index as df_test
    df_predicted_ukf = pd.DataFrame(predicted_means_ukf, index=df_test.index, columns=predict_cols)

    # Filter data for the specific hour
    # start_time = '2025-10-09 12:50:00' # Use the same time range as before
    # end_time = '2025-10-09 12:59:59'

    # df_test_hour = df_test.loc[start_time:end_time]
    # df_predicted_ukf_hour = df_predicted_ukf.loc[start_time:end_time]

    return df_predicted_ukf

def experiment_kf(df_train, df_test, predict_cols, n_latent_variables = 10):
    # Define the number of latent variables and observation dimensions
    # n_latent_variables = 10
    # predict_cols = ['basis1_dt', 'basis2_dt', 'spot_bids_slope_d', 'spot_asks_slope_d', 'swap_bids_slope_d', 'swap_asks_slope_d']
    n_observation_dimensions = len(predict_cols) # We are observing basis1, basis2, basis1_dt, and basis2_dt

    # Initialize Kalman Filter with 4 latent variables and 4 observation dimensions
    kf = KalmanFilter(transition_matrices=np.eye(n_latent_variables), # Identity matrix for state transition
                      observation_matrices=np.ones((n_observation_dimensions, n_latent_variables)), # Observe a linear combination of states for all four observations
                      initial_state_mean=np.zeros(n_latent_variables), # Initial state mean (zeros)
                      initial_state_covariance=np.eye(n_latent_variables), # Initial state covariance (identity)
                      observation_covariance=np.eye(n_observation_dimensions), # Observation noise covariance (identity matrix for 4D observation)
                      transition_covariance=np.eye(n_latent_variables) * 0.01) # Transition noise covariance

    # print(df_train.info())
    # Train the Kalman filter on the training data
    state_means, state_covariances = kf.filter(df_train[predict_cols].values)


    # Predict the basis1 and basis2 for the test period using the trained Kalman filter, updating with previous actual observation
    last_state_mean = state_means[-1]
    last_state_covariance = state_covariances[-1]

    predicted_means = []
    predicted_covariances = []

    # Ensure current_mean and current_covariance have the correct shape
    current_mean = np.array(last_state_mean).reshape(-1, 1)
    current_covariance = np.array(last_state_covariance)


    # Convert transition_matrices[0], observation_matrices[0] and transition_covariance[0] to numpy arrays
    transition_matrix = np.array(kf.transition_matrices)
    observation_matrix = np.array(kf.observation_matrices)
    transition_covariance_matrix = np.array(kf.transition_covariance)
    observation_covariance_matrix = np.array(kf.observation_covariance) # Get observation covariance matrix


    # Predict step by step for the length of the test data
    for i in range(len(df_test)):
        # Predict the next state
        predicted_state_mean = transition_matrix @ current_mean
        predicted_state_covariance = transition_matrix @ current_covariance @ transition_matrix.T + transition_covariance_matrix

        # If not the first time step, use the previous actual observation to update the state
        if i > 0:
            # Get the actual observation from df_test for the previous time step
            previous_actual_observation = df_test[predict_cols].iloc[i-1].values.reshape(-1, 1)

            # Calculate the Kalman Gain
            innovation_covariance = observation_matrix @ predicted_state_covariance @ observation_matrix.T + observation_covariance_matrix
            kalman_gain = predicted_state_covariance @ observation_matrix.T @ np.linalg.inv(innovation_covariance)

            # Update the state using the previous actual observation
            current_mean = predicted_state_mean + kalman_gain @ (previous_actual_observation - observation_matrix @ predicted_state_mean)
            current_covariance = predicted_state_covariance - kalman_gain @ observation_matrix @ predicted_state_covariance
        else:
            # For the first time step, use the predicted state from the training period
            current_mean = predicted_state_mean
            current_covariance = predicted_state_covariance


        # The prediction of the observation is based on the current (updated) state
        predicted_observation_mean = observation_matrix @ current_mean

        predicted_means.append(predicted_observation_mean.flatten()) # Append the 4D predicted observation
        # We are not predicting observation covariance in this loop, but state covariance
        predicted_covariances.append(current_covariance.diagonal()) # Append diagonal of the state covariance


    # Convert predicted means to a numpy array
    predicted_means = np.array(predicted_means)


    # Create a DataFrame for the predicted values with the same index as df_test
    df_predicted_kf = pd.DataFrame(predicted_means, index=df_test.index, columns=predict_cols)

    return df_predicted_kf

def experiment_pf(df_train, df_test, predict_cols, n_latent_variables = 10):

    # Define the number of particles
    n_particles = 50
    
    n_observation_dimensions = len(predict_cols) # We are observing basis1, basis2, basis1_dt, and basis2_dt

    # Define the state transition function (non-linear example)
    def state_transition(x, dt=1):
        """
        Non-linear state transition function.
        x: current state (numpy array of size n_latent_variables)
        dt: time step
        """
        # Example: Simple non-linear transition with some noise
        # This should be adapted to your specific problem's dynamics
        # For simplicity, let's assume a random walk with some non-linearity
        return x + normal(0, 0.001, size=x.shape) + 0.01 * np.sin(x * dt)

    # Define the observation function (non-linear example)
    def observation_function_pf(x):
        """
        Non-linear observation function.
        x: current state (numpy array of size n_latent_variables)
        """
        # Example: Observe a non-linear combination of the state
        # For simplicity, let's assume we observe the first n_observation_dimensions states
        # with some non-linearity and noise
        return x[:n_observation_dimensions] + normal(0, 0.001, size=n_observation_dimensions)

    # Define the likelihood function
    def likelihood(observation, predicted_observation, R):
        """
        Calculate the likelihood of the observation given the predicted observation.
        observation: actual observation (numpy array)
        predicted_observation: predicted observation from the particle (numpy array)
        R: observation noise covariance (scalar or matrix)
        """
        # Assuming Gaussian likelihood for simplicity
        # This calculates the probability density of the observation given the predicted observation
        # based on the observation noise covariance R.
        # For a multi-variate Gaussian, this involves the determinant and inverse of R.
        # For simplicity with scalar R, we can use the formula for a univariate Gaussian.
        # You will need to adapt this for a multi-variate observation and covariance matrix R.

        # For a multi-variate Gaussian likelihood:
        from scipy.stats import multivariate_normal
        return multivariate_normal.pdf(observation, mean=predicted_observation, cov=R)


    # Initialize particles randomly
    initial_state_range = [-0.1, 0.1] # Define a reasonable range for initial states based on your data

    # Initialize particles with random values within the defined range
    particles = uniform(initial_state_range[0], initial_state_range[1], size=(n_particles, n_latent_variables))
    weights = np.ones(n_particles) / n_particles # Initialize weights uniformly

    # Lists to store predicted means for plotting
    predicted_means_pf = []

    # Process each time step in the test data
    for i in range(len(df_test)):
        # Predict the next state for each particle
        for j in range(n_particles):
            particles[j, :] = state_transition(particles[j, :])

        # Get the actual observation for the current time step
        # Note: In a real-world scenario, you would only have the observation
        # up to the current time step to predict the next one.
        # For this simulation, we use the actual observation from the test set
        # to update the weights.
        actual_observation = df_test[predict_cols].iloc[max(i-1, 0)].values

        # Update weights based on the likelihood of the observation
        for j in range(n_particles):
            predicted_observation = observation_function_pf(particles[j, :])
            # Define observation noise covariance R (example: identity matrix * small value)
            R = np.eye(n_observation_dimensions) * 0.001 # Adjust this based on expected observation noise
            weights[j] *= likelihood(actual_observation, predicted_observation, R)

        # Normalize weights
        weights += 1.e-300 # Add a small number to avoid division by zero
        weights /= sum(weights)

        # Resample particles
        indices = systematic_resample(weights)
        particles[:] = particles[indices]
        weights[:] = np.ones(n_particles) / n_particles # Reset weights after resampling

        # Estimate the state mean (e.g., weighted average of particles)
        estimated_state_mean = np.average(particles, weights=weights, axis=0)

        # Predict the observation based on the estimated state mean
        predicted_observation_mean_pf = observation_function_pf(estimated_state_mean)

        # Append the predicted observation mean
        # We are predicting the observation based on the estimated state
        # Since our observation function is simplified to return the first n_observation_dimensions
        # of the state, the predicted observation mean will be the first n_observation_dimensions
        # of the estimated state mean.
        predicted_means_pf.append(estimated_state_mean[:n_observation_dimensions])


    # Convert predicted means to a numpy array
    predicted_means_pf = np.array(predicted_means_pf)

    # Create a DataFrame for the predicted values with the same index as df_test
    df_predicted_pf = pd.DataFrame(predicted_means_pf, index=df_test.index, columns=predict_cols)

    return df_predicted_pf
