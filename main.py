import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import csv
import tensorflow as tf

# --- Part 1: Numerical Solver (to generate ground truth data) ---
# This part is framework-agnostic and remains unchanged.

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
        self.Nn_scale = 1.0e+08 # Use max value from IC
        self.Nt_scale = 1.0e+07
        self.Ch_scale = 2.5e-07
        self.t_scale = 5.0e+06
        self.r_scale = 0.5


def numerical_pde_system(t, u, params):
    """Defines the ODE system for the numerical solver."""
    params.ncall += 1
    n = params.n
    Nn, Nt, Ch = u[0:n], u[n:2*n], u[2*n:3*n]
    
    Nnr, Ntr, Chr = np.gradient(Nn, params.dr), np.gradient(Nt, params.dr), np.gradient(Ch, params.dr)
    Ntr[0], Ntr[-1] = 0, 0
    Chr[0], Chr[-1] = 0, 0
    
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
        self.hidden = [tf.keras.layers.Dense(units, activation='tanh', kernel_initializer='glorot_normal') for units in layers[1:-1]]
        self.output_layer = tf.keras.layers.Dense(layers[-1], kernel_initializer='glorot_normal', activation=None)

    def call(self, inputs):
        t, r = inputs[:, 0:1], inputs[:, 1:2]
        t_norm = 2.0 * (t / self.params.t_scale) - 1.0
        r_norm = 2.0 * (r / self.params.r_scale) - 1.0
        x = tf.concat([t_norm, r_norm], axis=1)
        for layer in self.hidden:
            x = layer(x)
        u_scaled = self.output_layer(x)
        return u_scaled

def get_pde_residuals(model, t, r, params):
    """Calculates PDE residuals using tf.GradientTape."""
    epsilon = 1e-8
    t_tensor = tf.convert_to_tensor(t, dtype=tf.float32)
    r_tensor = tf.convert_to_tensor(r, dtype=tf.float32)
    
    # --- FIX: Use correct nested GradientTape structure for second derivatives ---
    with tf.GradientTape(persistent=True) as outer_tape:
        outer_tape.watch(t_tensor)
        outer_tape.watch(r_tensor)
        with tf.GradientTape(persistent=True) as inner_tape:
            inner_tape.watch(t_tensor)
            inner_tape.watch(r_tensor)
            
            inputs = tf.concat([t_tensor, r_tensor], axis=1)
            u_scaled = model(inputs)

        # First derivatives are calculated with the inner tape
        u_scaled_t = inner_tape.gradient(u_scaled, t_tensor)
        u_scaled_r = inner_tape.gradient(u_scaled, r_tensor)

    # Second derivative (d/dr of du/dr) is calculated with the outer tape
    u_scaled_rr = outer_tape.gradient(u_scaled_r, r_tensor)

    # Clean up persistent tapes
    del inner_tape
    del outer_tape
    
    if u_scaled_t is None or u_scaled_r is None or u_scaled_rr is None:
        raise ValueError("Gradient calculation returned None. Check the graph connections.")

    Nnt = u_scaled_t[:, 0:1] * params.Nn_scale / params.t_scale
    Ntt = u_scaled_t[:, 1:2] * params.Nt_scale / params.t_scale
    Cht = u_scaled_t[:, 2:3] * params.Ch_scale / params.t_scale

    Nnr = u_scaled_r[:, 0:1] * params.Nn_scale / params.r_scale
    Ntr = u_scaled_r[:, 1:2] * params.Nt_scale / params.r_scale
    Chr = u_scaled_r[:, 2:3] * params.Ch_scale / params.r_scale

    Nnrr = u_scaled_rr[:, 0:1] * params.Nn_scale / (params.r_scale**2)
    Ntrr = u_scaled_rr[:, 1:2] * params.Nt_scale / (params.r_scale**2)
    Chrr = u_scaled_rr[:, 2:3] * params.Ch_scale / (params.r_scale**2)

    Nn = u_scaled[:, 0:1] * params.Nn_scale
    Nt = u_scaled[:, 1:2] * params.Nt_scale
    Ch = u_scaled[:, 2:3] * params.Ch_scale

    f_Nn = Nnt - (params.rn1 * Nn * (1 - Nn / params.Kn) - params.rn2 * Ch * Nn)
    D = params.Dt * (1 - Nn / params.Kn)
    f_Nt = Ntt - (params.rt1 * Nt * (1 - Nt / params.Kt) + D * (Ntrr + 2/(r_tensor + epsilon) * Ntr) + (-params.Dt/params.Kn) * Nnr * Ntr)
    f_Ch = Cht - (params.rh1 * Nt - params.rh2 * Ch + params.Dh * (Chrr + 2/(r_tensor + epsilon) * Chr))
    
    f_Nt_r0 = Ntt - (params.rt1 * Nt * (1 - Nt / params.Kt) + 3 * D * Ntrr)
    f_Ch_r0 = Cht - (params.rh1 * Nt - params.rh2 * Ch + 3 * params.Dh * Chrr)
    
    f_Nt = tf.where(r_tensor < 1e-6, f_Nt_r0, f_Nt)
    f_Ch = tf.where(r_tensor < 1e-6, f_Ch_r0, f_Ch)

    return f_Nn, f_Nt, f_Ch

def get_bc_residuals(model, t_bc, r_bc, params):
    """Calculates boundary condition residuals."""
    with tf.GradientTape() as tape:
        tape.watch(r_bc)
        inputs = tf.concat([t_bc, r_bc], axis=1)
        u_scaled = model(inputs)
    
    u_scaled_r = tape.gradient(u_scaled, r_bc)
    
    Ntr_bc_scaled = u_scaled_r[:, 1:2] * params.Nt_scale / params.r_scale
    Chr_bc_scaled = u_scaled_r[:, 2:3] * params.Ch_scale / params.r_scale

    return Ntr_bc_scaled, Chr_bc_scaled

def run_pinn_training_tf():
    print(f"--- Starting PINN Training with TensorFlow ---")
    params = ModelParameters()
    t_data, r_data, Nn_clean, Nt_clean, Ch_clean, Nn_noisy, Nt_noisy, Ch_noisy = generate_ground_truth_data(params)
    
    t_flat = np.tile(t_data, (params.n, 1)).T.flatten()[:, np.newaxis]
    r_flat = np.tile(r_data, (len(t_data), 1)).flatten()[:, np.newaxis]
    
    X_data = tf.constant(np.hstack([t_flat, r_flat]), dtype=tf.float32)
    U_data_scaled = tf.constant(np.hstack([
        Nn_noisy.flatten()[:,np.newaxis] / params.Nn_scale,
        Nt_noisy.flatten()[:,np.newaxis] / params.Nt_scale,
        Ch_noisy.flatten()[:,np.newaxis] / params.Ch_scale
    ]), dtype=tf.float32)
    
    u0_full = get_initial_conditions(params)
    X_ic = tf.constant(np.hstack([np.zeros_like(params.r)[:,np.newaxis], params.r[:,np.newaxis]]), dtype=tf.float32)
    U_ic_scaled = tf.constant(np.hstack([
        u0_full[0:params.n][:,np.newaxis] / params.Nn_scale,
        u0_full[params.n:2*params.n][:,np.newaxis] / params.Nt_scale,
        u0_full[2*params.n:3*params.n][:,np.newaxis] / params.Ch_scale
    ]), dtype=tf.float32)
    
    t_bc_np = np.linspace(0, 5.0e+06, 100)[:, np.newaxis]
    X_bc_r0 = tf.constant(np.hstack([t_bc_np, np.full_like(t_bc_np, params.rl)]), dtype=tf.float32)
    X_bc_ru = tf.constant(np.hstack([t_bc_np, np.full_like(t_bc_np, params.ru)]), dtype=tf.float32)
    
    N_collocation = 10000
    t_colloc = tf.random.uniform((N_collocation, 1), 0, 5.0e+06)
    r_colloc = tf.random.uniform((N_collocation, 1), 0, 0.5)

    layers = [2] + [64] * 5 + [3]
    pinn_net = PINN(layers, params)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps=2000, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            u_pred_scaled = pinn_net(X_data)
            loss_data = loss_fn(u_pred_scaled, U_data_scaled)
            
            u_ic_pred_scaled = pinn_net(X_ic)
            loss_ic = loss_fn(u_ic_pred_scaled, U_ic_scaled)
            
            f_Nn, f_Nt, f_Ch = get_pde_residuals(pinn_net, t_colloc, r_colloc, params)
            loss_pde = loss_fn(f_Nn / params.Nn_scale, 0) + loss_fn(f_Nt / params.Nt_scale, 0) + loss_fn(f_Ch / params.Ch_scale, 0)
            
            Ntr_bc_r0, Chr_bc_r0 = get_bc_residuals(pinn_net, X_bc_r0[:, 0:1], X_bc_r0[:, 1:2], params)
            Ntr_bc_ru, Chr_bc_ru = get_bc_residuals(pinn_net, X_bc_ru[:, 0:1], X_bc_ru[:, 1:2], params)
            loss_bc = loss_fn(Ntr_bc_r0, 0) + loss_fn(Chr_bc_r0, 0) + loss_fn(Ntr_bc_ru, 0) + loss_fn(Chr_bc_ru, 0)
            
            total_loss = 20 * loss_data + 20 * loss_ic +50 * loss_pde + 50* loss_bc
            
        grads = tape.gradient(total_loss, pinn_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, pinn_net.trainable_variables))
        return total_loss, loss_data, loss_ic, loss_pde, loss_bc

    epochs = 5000
    for epoch in range(epochs):
        total_loss, loss_data, loss_ic, loss_pde, loss_bc = train_step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Total Loss: {total_loss.numpy():.4e}, "
                  f"Data: {loss_data.numpy():.4e}, IC: {loss_ic.numpy():.4e}, "
                  f"PDE: {loss_pde.numpy():.4e}, BC: {loss_bc.numpy():.4e}")
            
    print("\n--- Evaluating Trained PINN Model ---")
    u_pinn_scaled = pinn_net(X_data).numpy()
    u_pinn_pred = np.hstack([
        u_pinn_scaled[:, 0:1] * params.Nn_scale,
        u_pinn_scaled[:, 1:2] * params.Nt_scale,
        u_pinn_scaled[:, 2:3] * params.Ch_scale,
    ]).reshape(len(t_data), params.n, 3)
    
    Nn_pinn, Nt_pinn, Ch_pinn = u_pinn_pred[:,:,0], u_pinn_pred[:,:,1], u_pinn_pred[:,:,2]

    l2_error_Nn = np.linalg.norm(Nn_clean - Nn_pinn) / np.linalg.norm(Nn_clean)
    l2_error_Nt = np.linalg.norm(Nt_clean - Nt_pinn) / np.linalg.norm(Nt_clean)
    l2_error_Ch = np.linalg.norm(Ch_clean - Ch_pinn) / np.linalg.norm(Ch_clean)
    print(f"Relative L2 Error (Nn): {l2_error_Nn:.4e}")
    print(f"Relative L2 Error (Nt): {l2_error_Nt:.4e}")
    print(f"Relative L2 Error (Ch): {l2_error_Ch:.4e}")

    time_indices_to_plot = [0, 5, 10, 15, 20]
    fig, axs = plt.subplots(3, len(time_indices_to_plot), figsize=(20, 12), sharey='row')
    fig.suptitle('PINN vs. Numerical Solution Comparison (Training on Noisy Data)', fontsize=16)

    for i, t_idx in enumerate(time_indices_to_plot):
        t_days = t_data[t_idx] / (60 * 60 * 24)
        
        axs[0, i].set_title(f't = {t_days:.1f} days')
        # Plot the original clean data as a line
        axs[0, i].plot(r_data, Nn_clean[t_idx, :], 'b-', label='Numerical (Clean)')
        # Plot the noisy data the PINN was trained on as dots
        axs[0, i].plot(r_data, Nn_noisy[t_idx, :], 'k.', markersize=2, label='Training Data (Noisy)')
        # Plot the PINN's prediction
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
    plt.show()

if __name__ == '__main__':
    run_pinn_training_tf()
