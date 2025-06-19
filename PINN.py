import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import csv
import tensorflow as tf
import os

# --- Part 1: Numerical Solver (to generate ground truth data) ---

class ModelParameters:
    def __init__(self):
        # Grid and model parameters
        self.rl, self.ru, self.n = 0.0, 0.5, 101
        self.r = np.linspace(self.rl, self.ru, self.n)
        self.dr = self.r[1] - self.r[0]
        self.rn1, self.rn2, self.Kn = 1.0e-06, 1.0, 5.0e+07
        self.rt1, self.Dt, self.Kt = 1.0e-06, 2.0e-10, 5.0e+07
        self.rh1, self.rh2, self.Dh = 2.2e-17, 1.1e-04, 5.0e-06
        self.ncall = 0

        # Characteristic scales for normalization
        self.Nn_scale = 1.0e+08
        self.Nt_scale = 1.0e+07
        self.Ch_scale = 1.0e-07
        self.t_scale = 5.0e+06
        self.r_scale = 0.5

        # Add minimum radius to avoid division by zero
        self.r_min = 1e-8


def numerical_pde_system(t, u, params):
    """Defines the ODE system for the numerical solver."""
    params.ncall += 1
    n = params.n
    Nn, Nt, Ch = u[0:n], u[n:2*n], u[2*n:3*n]

    Nnr, Ntr, Chr = np.gradient(Nn, params.dr), np.gradient(Nt, params.dr), np.gradient(Ch, params.dr)
    Ntr[0], Ntr[-1] = 0, 0  # Boundary conditions
    Chr[0], Chr[-1] = 0, 0  # Boundary conditions

    Ntrr, Chrr = np.gradient(Ntr, params.dr), np.gradient(Chr, params.dr)

    Nnt, Ntt, Cht = np.zeros(n), np.zeros(n), np.zeros(n)

    for i in range(n):
        D = params.Dt * (1 - Nn[i] / params.Kn)
        if D < 0: D = 0

        Nnt[i] = params.rn1 * Nn[i] * (1 - Nn[i] / params.Kn) - params.rn2 * Ch[i] * Nn[i]
        if i == 0:
            Ntt[i] = params.rt1 * Nt[i] * (1 - Nt[i] / params.Kt) + 3 * D * Ntrr[i]
            Cht[i] = params.rh1 * Nt[i] - params.rh2 * Ch[i] + 3 * params.Dh * Chrr[i]
        else:
            diffusion_Nt = D * (Ntrr[i] + 2/params.r[i] * Ntr[i]) + (-params.Dt/params.Kn) * Nnr[i] * Ntr[i]
            Ntt[i] = params.rt1 * Nt[i] * (1 - Nt[i] / params.Kt) + diffusion_Nt
            diffusion_Ch = params.Dh * (Chrr[i] + 2/params.r[i] * Chr[i])
            Cht[i] = params.rh1 * Nt[i] - params.rh2 * Ch[i] + diffusion_Ch

    return np.concatenate((Nnt, Ntt, Cht))

def get_initial_conditions(params):
    """Generates the initial conditions for both numerical and PINN models."""
    r, n = params.r, params.n
    u0 = np.zeros(3 * n)
    rs = 50
    r_transition_cells, r_transition_H = r[20], r[10]

    tanhr_Nn = np.tanh(rs * (r - r_transition_cells))
    u0[0:n] = 5.0e+07 * (1 - tanhr_Nn) / 2 + 1.0e+08 * (1 + tanhr_Nn) / 2

    tanhr_Nt = np.tanh(rs * (r - r_transition_cells))
    u0[n:2*n] = 1.0e+05 * (1 - tanhr_Nt) / 2 + 1.0e+03 * (1 + tanhr_Nt) / 2

    tanhr_Ch = np.tanh(rs * (r - r_transition_H))
    u0[2*n:3*n] = 1.0e-09 * (1 - tanhr_Ch) / 2
    return u0

def generate_ground_truth_data(params):
    """Runs the numerical solver and saves the data to a CSV file."""
    print("--- Running Numerical Solver to Generate Ground Truth Data ---")
    t0, tf, nout = 0.0, 5.0e+06, 21
    tout = np.linspace(t0, tf, nout)
    u0 = get_initial_conditions(params)

    sol = solve_ivp(numerical_pde_system, [t0, tf], u0, method='BDF', t_eval=tout, args=(params,))

    if not sol.success:
        raise RuntimeError("Numerical ODE solver failed:", sol.message)

    u = sol.y.T
    Nn, Nt, Ch = u[:, :params.n], u[:, params.n:2*params.n], u[:, 2*params.n:3*params.n]

    noise_level = 0.01 # 1% noise
    Nn_noisy = Nn + noise_level * np.std(Nn) * np.random.randn(*Nn.shape)
    Nt_noisy = Nt + noise_level * np.std(Nt) * np.random.randn(*Nt.shape)
    Ch_noisy = Ch + noise_level * np.std(Ch) * np.random.randn(*Ch.shape)

    csv_filename = 'tumor_growth_data.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'r', 'Nn', 'Nt', 'Ch'])
        for it, t_val in enumerate(tout):
            for i, r_val in enumerate(params.r):
                writer.writerow([t_val, r_val, Nn_noisy[it, i], Nt_noisy[it, i], Ch_noisy[it, i]])
    print(f"Noisy ground truth data saved to {csv_filename}\n")
    return tout, params.r, Nn, Nt, Ch, Nn_noisy, Nt_noisy, Ch_noisy

# --- Part 2: Physics-Informed Neural Network (PINN) using TensorFlow ---

class PINN(tf.keras.Model):
    """Defines the PINN architecture using TensorFlow/Keras."""
    def __init__(self, layers, params):
        super(PINN, self).__init__()
        self.params = params
        self.hidden = []
        for units in layers[1:-1]:
            layer = tf.keras.layers.Dense(
                units,
                activation='tanh',
                kernel_initializer='glorot_normal',
                bias_initializer='zeros'
            )
            self.hidden.append(layer)

        self.output_layer = tf.keras.layers.Dense(
            layers[-1],
            kernel_initializer='glorot_normal',
            bias_initializer='zeros',
            activation=None
        )

    def call(self, inputs):
        t, r = inputs[:, 0:1], inputs[:, 1:2]
        t_norm = t / self.params.t_scale
        r_norm = r / self.params.r_scale
        x = tf.concat([t_norm, r_norm], axis=1)

        for layer in self.hidden:
            x = layer(x)
        u_raw = self.output_layer(x)

        # Apply sigmoid activation to ensure positive outputs
        u_scaled = tf.nn.sigmoid(u_raw)
        return u_scaled

def get_pde_residuals(model, t, r, params):
    """Calculates PDE residuals using tf.GradientTape with improved stability."""
    epsilon = 1e-8
    t_tensor = tf.convert_to_tensor(t, dtype=tf.float32)
    r_tensor = tf.convert_to_tensor(r, dtype=tf.float32)

    # Ensure r is never exactly zero
    r_tensor = tf.maximum(r_tensor, epsilon)

    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([t_tensor, r_tensor])
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([t_tensor, r_tensor])

            inputs = tf.concat([t_tensor, r_tensor], axis=1)
            u_scaled = model(inputs)

            # Scale back to physical units
            Nn = u_scaled[:, 0:1] * params.Nn_scale
            Nt = u_scaled[:, 1:2] * params.Nt_scale
            Ch = u_scaled[:, 2:3] * params.Ch_scale

        # First derivatives
        Nn_t = tape2.gradient(Nn, t_tensor)
        Nt_t = tape2.gradient(Nt, t_tensor)
        Ch_t = tape2.gradient(Ch, t_tensor)

        Nn_r = tape2.gradient(Nn, r_tensor)
        Nt_r = tape2.gradient(Nt, r_tensor)
        Ch_r = tape2.gradient(Ch, r_tensor)

    # Second derivatives
    Nt_rr = tape1.gradient(Nt_r, r_tensor)
    Ch_rr = tape1.gradient(Ch_r, r_tensor)

    # Clean up tapes
    del tape1, tape2

    # Check for None gradients and handle them
    if Nn_t is None: Nn_t = tf.zeros_like(Nn)
    if Nt_t is None: Nt_t = tf.zeros_like(Nt)
    if Ch_t is None: Ch_t = tf.zeros_like(Ch)
    if Nn_r is None: Nn_r = tf.zeros_like(Nn)
    if Nt_r is None: Nt_r = tf.zeros_like(Nt)
    if Ch_r is None: Ch_r = tf.zeros_like(Ch)
    if Nt_rr is None: Nt_rr = tf.zeros_like(Nt)
    if Ch_rr is None: Ch_rr = tf.zeros_like(Ch)

    # Add small epsilon to prevent division by zero
    Nn = tf.maximum(Nn, epsilon)
    Nt = tf.maximum(Nt, epsilon)
    Ch = tf.maximum(Ch, epsilon)

    # PDE residuals with improved numerical stability
    f_Nn = Nn_t - (params.rn1 * Nn * (1 - Nn / params.Kn) - params.rn2 * Ch * Nn)

    # Diffusion coefficient with better bounds
    D = tf.maximum(params.Dt * (1 - Nn / params.Kn), 0.0)

    # Handle r=0 case more carefully
    r_safe = tf.maximum(r_tensor, epsilon)

    # Diffusion terms with safer computation
    diffusion_Nt = D * (Nt_rr + 2.0 / r_safe * Nt_r)
    diffusion_Ch = params.Dh * (Ch_rr + 2.0 / r_safe * Ch_r)

    # Special case for r ≈ 0 (use L'Hôpital's rule result)
    is_near_zero = tf.less(r_tensor, epsilon * 100)
    diffusion_Nt_r0 = 3.0 * D * Nt_rr
    diffusion_Ch_r0 = 3.0 * params.Dh * Ch_rr

    diffusion_Nt = tf.where(is_near_zero, diffusion_Nt_r0, diffusion_Nt)
    diffusion_Ch = tf.where(is_near_zero, diffusion_Ch_r0, diffusion_Ch)

    f_Nt = Nt_t - (params.rt1 * Nt * (1 - Nt / params.Kt) + diffusion_Nt)
    f_Ch = Ch_t - (params.rh1 * Nt - params.rh2 * Ch + diffusion_Ch)

    return f_Nn, f_Nt, f_Ch

def safe_gradient(model, X, output_index):
    """Helper function to safely compute gradients with None checks."""
    with tf.GradientTape() as tape:
        tape.watch(X)
        u = model(X)
        output = u[:, output_index:output_index+1]
    grad = tape.gradient(output, X)
    if grad is None:
        return tf.zeros_like(X)
    return grad

def run_pinn_training_tf():
    print(f"--- Starting PINN Training with TensorFlow ---")
    params = ModelParameters()
    t_data, r_data, Nn_clean, Nt_clean, Ch_clean, Nn_noisy, Nt_noisy, Ch_noisy = generate_ground_truth_data(params)

    # Prepare training data
    t_flat = np.tile(t_data, (params.n, 1)).T.flatten()[:, np.newaxis]
    r_flat = np.tile(r_data, (len(t_data), 1)).flatten()[:, np.newaxis]

    X_data = tf.constant(np.hstack([t_flat, r_flat]), dtype=tf.float32)

    # Scale to [0,1] range for sigmoid output
    U_data_scaled = tf.constant(np.hstack([
        Nn_noisy.flatten()[:,np.newaxis] / params.Nn_scale,
        Nt_noisy.flatten()[:,np.newaxis] / params.Nt_scale,
        Ch_noisy.flatten()[:,np.newaxis] / params.Ch_scale
    ]), dtype=tf.float32)

    # Initial conditions
    u0_full = get_initial_conditions(params)
    X_ic = tf.constant(np.hstack([np.zeros_like(params.r)[:,np.newaxis], params.r[:,np.newaxis]]), dtype=tf.float32)
    U_ic_scaled = tf.constant(np.hstack([
        u0_full[0:params.n][:,np.newaxis] / params.Nn_scale,
        u0_full[params.n:2*params.n][:,np.newaxis] / params.Nt_scale,
        u0_full[2*params.n:3*params.n][:,np.newaxis] / params.Ch_scale
    ]), dtype=tf.float32)

    # Boundary conditions - left (r=0) and right (r=0.5)
    # Left boundary (r=0)
    t_left = tf.random.uniform((500, 1), 0, 5.0e+06, dtype=tf.float32)
    r_left = tf.zeros_like(t_left) + params.r_min  # Small value near zero
    X_left = tf.concat([t_left, r_left], axis=1)

    # Right boundary (r=0.5)
    t_right = tf.random.uniform((500, 1), 0, 5.0e+06, dtype=tf.float32)
    r_right = tf.ones_like(t_right) * 0.5
    X_right = tf.concat([t_right, r_right], axis=1)

    # Collocation points
    N_collocation = 2000
    t_colloc = tf.random.uniform((N_collocation, 1), 0, 5.0e+06, dtype=tf.float32)
    r_colloc = tf.random.uniform((N_collocation, 1), params.r_min, 0.5, dtype=tf.float32)

    # Network architecture
    layers = [2, 80, 80, 80, 3]
    pinn_net = PINN(layers, params)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            # Data loss
            u_pred_scaled = pinn_net(X_data)
            loss_data = tf.reduce_mean(tf.square(u_pred_scaled - U_data_scaled))

            # Initial condition loss
            u_ic_pred_scaled = pinn_net(X_ic)
            loss_ic = tf.reduce_mean(tf.square(u_ic_pred_scaled - U_ic_scaled))

            # Boundary condition losses
            # Left boundary (r=0): zero gradient condition
            Nt_r_left = safe_gradient(pinn_net, X_left, 1)[:, 1:2]  # dNt/dr at r=0
            Ch_r_left = safe_gradient(pinn_net, X_left, 2)[:, 1:2]  # dCh/dr at r=0

            # Right boundary (r=0.5): zero gradient condition
            Nt_r_right = safe_gradient(pinn_net, X_right, 1)[:, 1:2]  # dNt/dr at r=0.5
            Ch_r_right = safe_gradient(pinn_net, X_right, 2)[:, 1:2]  # dCh/dr at r=0.5

            loss_bc = (tf.reduce_mean(tf.square(Nt_r_left)) +
                       tf.reduce_mean(tf.square(Ch_r_left)) +
                       tf.reduce_mean(tf.square(Nt_r_right)) +
                       tf.reduce_mean(tf.square(Ch_r_right)))

            # PDE loss
            f_Nn, f_Nt, f_Ch = get_pde_residuals(pinn_net, t_colloc, r_colloc, params)
            loss_pde = (tf.reduce_mean(tf.square(f_Nn)) +
                        tf.reduce_mean(tf.square(f_Nt)) +
                        tf.reduce_mean(tf.square(f_Ch)))

            # Combined loss with weighting
            total_loss = (500 * loss_data +
                          50 * loss_ic +
                          50 * loss_bc +
                           1 * loss_pde)

        gradients = tape.gradient(total_loss, pinn_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pinn_net.trainable_variables))
        return total_loss, loss_data, loss_ic, loss_bc, loss_pde

    # Training loop
    epochs = 10000

    for epoch in range(epochs):
        total_loss, loss_data, loss_ic, loss_bc, loss_pde = train_step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Total Loss: {total_loss.numpy():.4e}, "
                  f"Data: {loss_data.numpy():.4e}, IC: {loss_ic.numpy():.4e}, "
                  f"BC: {loss_bc.numpy():.4e}, PDE: {loss_pde.numpy():.4e}")

    print("\n--- Evaluating Trained PINN Model ---")
    u_pinn_scaled = pinn_net(X_data).numpy()
    u_pinn_pred = np.hstack([
        u_pinn_scaled[:, 0:1] * params.Nn_scale,
        u_pinn_scaled[:, 1:2] * params.Nt_scale,
        u_pinn_scaled[:, 2:3] * params.Ch_scale,
    ]).reshape(len(t_data), params.n, 3)

    Nn_pinn, Nt_pinn, Ch_pinn = u_pinn_pred[:,:,0], u_pinn_pred[:,:,1], u_pinn_pred[:,:,2]

    # Calculate errors
    def safe_l2_error(pred, true):
        """Calculates the relative L2 error, avoiding division by zero."""
        diff_norm = np.linalg.norm(true - pred)
        true_norm = np.linalg.norm(true)
        return diff_norm / max(true_norm, 1e-10)

    l2_error_Nn = safe_l2_error(Nn_pinn, Nn_clean)
    l2_error_Nt = safe_l2_error(Nt_pinn, Nt_clean)
    l2_error_Ch = safe_l2_error(Ch_pinn, Ch_clean)

    print(f"Relative L2 Error (Nn): {l2_error_Nn:.4e}")
    print(f"Relative L2 Error (Nt): {l2_error_Nt:.4e}")
    print(f"Relative L2 Error (Ch): {l2_error_Ch:.4e}")

    # --- EDITED SECTION: Calculate and Print Overall Accuracy ---
    # Average the relative errors from the three variables
    mean_relative_error = (l2_error_Nn + l2_error_Nt + l2_error_Ch) / 3.0

    # Calculate accuracy as 100% minus the average error percentage
    accuracy = (1.0 - mean_relative_error) * 100

    print("\n-------------------------------------")
    print(f"📊 Overall Model Accuracy: {accuracy:.2f}%")
    print("-------------------------------------")
    # --- END EDITED SECTION ---


    # Plotting
    time_indices_to_plot = [0, 5, 10, 15, 20]
    fig, axs = plt.subplots(3, len(time_indices_to_plot), figsize=(20, 12), sharey='row')
    fig.suptitle('PINN vs. Numerical Solution Comparison (Training on Noisy Data)', fontsize=16)

    for i, t_idx in enumerate(time_indices_to_plot):
        t_days = t_data[t_idx] / (60 * 60 * 24)

        axs[0, i].set_title(f't = {t_days:.1f} days')
        axs[0, i].plot(r_data, Nn_clean[t_idx, :], 'b-', label='Numerical (Clean)')
        axs[0, i].plot(r_data, Nn_noisy[t_idx, :], 'k.', markersize=2, label='Training Data (Noisy)')
        axs[0, i].plot(r_data, Nn_pinn[t_idx, :], 'r--', label='PINN Prediction')
        if i == 0: axs[0, i].set_ylabel('Nn (normal cells)')

        axs[1, i].plot(r_data, Nt_clean[t_idx, :], 'b-')
        axs[1, i].plot(r_data, Nt_noisy[t_idx, :], 'k.', markersize=2)
        axs[1, i].plot(r_data, Nt_pinn[t_idx, :], 'r--')
        if i == 0: axs[1, i].set_ylabel('Nt (tumor cells)')

        axs[2, i].plot(r_data, Ch_clean[t_idx, :], 'b-')
        axs[2, i].plot(r_data, Ch_noisy[t_idx, :], 'k.', markersize=2)
        axs[2, i].plot(r_data, Ch_pinn[t_idx, :], 'r--')
        if i == 0: axs[2, i].set_ylabel('Ch (excess H+)')
        axs[2, i].set_xlabel('r (cm)')

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and show plot
    plot_filename = 'pinn_comparison.png'
    plt.savefig(plot_filename)
    plt.show()
    print(f"Plot saved as {plot_filename}")

if __name__ == '__main__':
    # Run the training
    run_pinn_training_tf()
