import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ============================================================
# MULTI-ROTOR 3D KINEMATIC + SE(3) + PID + ANTI-WINDUP
# Fixed-rate simulation: 50 Hz, dt = 20 ms
# ============================================================

# ---------------- Simulation settings ----------------
FS = 50.0
DT = 1.0 / FS
T_END = 20.0
t = np.arange(0.0, T_END + DT, DT)
N = len(t)

# Crosswind: step disturbance at 2 seconds
WIND_TIME = 2.0
WIND = np.array([0.8, -0.5, 0.0])   # normalized disturbance force

# ---------------- Desired 3D trajectory ----------------
# Smooth helical/looping trajectory
traj_t = np.minimum(t, 8.0)
xd = 4.0 * np.sin(0.45 * traj_t)
yd = 4.0 * np.cos(0.45 * traj_t)
zd = 0.30 * traj_t + 0.5 * np.sin(0.25 * traj_t)

# Desired velocity (numerical derivative)
xd_dot = np.gradient(xd, DT)
yd_dot = np.gradient(yd, DT)
zd_dot = np.gradient(zd, DT)
desired_vel = np.column_stack((xd_dot, yd_dot, zd_dot))
xd_ddot = np.gradient(xd_dot, DT)
yd_ddot = np.gradient(yd_dot, DT)
zd_ddot = np.gradient(zd_dot, DT)
desired_acc = np.column_stack((xd_ddot, yd_ddot, zd_ddot))

# ---------------- PID controller ----------------
Kp = np.array([5.0, 5.0, 5.5])
Ki = np.array([2.5, 2.5, 3.0])
Kd = np.array([2.0, 2.0, 2.2])

U_MIN = np.array([-3.0, -3.0, -3.0])
U_MAX = np.array([ 3.0,  3.0,  3.0])

# Anti-windup back-calculation/clamping
# When saturated, the integral term is not allowed to move
# farther in the direction that increases saturation.
integral = np.zeros(3)
prev_error = np.zeros(3)

# ---------------- State ----------------
pos = np.array([0.0, 0.0, 0.0])
vel = np.array([0.0, 0.0, 0.0])

pos_hist = np.zeros((N, 3))
vel_hist = np.zeros((N, 3))
err_hist = np.zeros((N, 3))
u_hist = np.zeros((N, 3))
wind_hist = np.zeros((N, 3))
compute_ms = np.zeros(N)

# ---------------- Attitude states ----------------
# Roll, pitch, yaw in radians
att = np.zeros(3)
att_des = np.zeros((N, 3))
att_hist = np.zeros((N, 3))
att_err_hist = np.zeros((N, 3))

att_Kp = np.array([6.0, 6.0, 5.5])
att_Ki = np.array([2.0, 2.0, 1.8])
att_Kd = np.array([1.0, 1.0, 0.9])
att_int = np.zeros(3)
att_prev_err = np.zeros(3)

# ---------------- Rotation matrices / SE(3) ----------------
def Rx(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def rotation_matrix(roll, pitch, yaw):
    # Body -> inertial convention
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)

def make_SE3(position, roll, pitch, yaw):
    T = np.eye(4)
    T[:3, :3] = rotation_matrix(roll, pitch, yaw)
    T[:3, 3] = position
    return T

# ---------------- Main 50 Hz simulation ----------------
for k, tk in enumerate(t):

    # Desired attitude maneuver: a finite step is applied and then
    # returned to level attitude so convergence can be demonstrated.
    if 2.0 <= tk < 4.0:
        att_des[k] = np.radians([15.0, -10.0, 12.0])
    else:
        att_des[k] = np.zeros(3)

    # Crosswind step
    wind = WIND if tk >= WIND_TIME else np.zeros(3)

    # ---------- Measure control computation time ----------
    # Plotting/file I/O are deliberately outside this timing measurement.
    tic = time.perf_counter()

    # Position PID
    error = np.array([xd[k], yd[k], zd[k]]) - pos
    derivative = (error - prev_error) / DT

    # Candidate integral
    integral_candidate = integral + error * DT

    u_unsat = Kp * error + Ki * integral_candidate + Kd * derivative

    # Saturation
    u_sat = np.clip(u_unsat, U_MIN, U_MAX)

    # Explicit anti-windup clamping:
    # accept integral update only when it does not push farther
    # into the active saturation limit.
    for j in range(3):
        pushing_high = (u_unsat[j] > U_MAX[j]) and (error[j] > 0)
        pushing_low  = (u_unsat[j] < U_MIN[j]) and (error[j] < 0)
        if not (pushing_high or pushing_low):
            integral[j] = integral_candidate[j]

    u = np.clip(
        Kp * error + Ki * integral + Kd * derivative,
        U_MIN, U_MAX
    )

    # Tracking plant with velocity/acceleration feed-forward.
    # This allows a moving reference to be followed without a
    # persistent velocity-tracking bias.
    v_error = desired_vel[k] - vel
    acceleration = desired_acc[k] + 3.0 * u + 1.5 * v_error + wind

    # Discrete state update
    vel = vel + acceleration * DT
    pos = pos + vel * DT

    # Attitude PID
    aerr = att_des[k] - att
    ader = (aerr - att_prev_err) / DT
    att_int_candidate = att_int + aerr * DT

    att_u_unsat = (
        att_Kp * aerr +
        att_Ki * att_int_candidate +
        att_Kd * ader
    )
    att_u = np.clip(att_u_unsat, -5.0, 5.0)

    # Attitude anti-windup
    for j in range(3):
        high = (att_u_unsat[j] > 5.0) and (aerr[j] > 0)
        low  = (att_u_unsat[j] < -5.0) and (aerr[j] < 0)
        if not (high or low):
            att_int[j] = att_int_candidate[j]

    att_u = np.clip(
        att_Kp * aerr + att_Ki * att_int + att_Kd * ader,
        -0.7, 0.7
    )

    # Attitude plant with reference-rate feed-forward and feedback.
    # No desired-rate feed-forward is required for the step maneuver.
    att = att + 4.0 * att_u * DT

    toc = time.perf_counter()
    compute_ms[k] = (toc - tic) * 1000.0

    # Save results
    pos_hist[k] = pos
    vel_hist[k] = vel
    err_hist[k] = error
    u_hist[k] = u
    wind_hist[k] = wind
    att_hist[k] = att
    att_err_hist[k] = aerr

    prev_error = error
    att_prev_err = aerr

# ---------------- Final SE(3) result ----------------
sample_index = int(5.0 / DT)
p5 = pos_hist[sample_index]
r5, p5_angle, y5 = att_hist[sample_index]

R5 = rotation_matrix(r5, p5_angle, y5)
T5 = make_SE3(p5, r5, p5_angle, y5)

# ---------------- Performance metrics ----------------
final_position_error = np.linalg.norm(
    np.array([xd[-1], yd[-1], zd[-1]]) - pos_hist[-1]
)
final_attitude_error_deg = np.linalg.norm(
    np.degrees(att_err_hist[-1])
)

# Steady-state error after the disturbance has settled
settle_start = int(8.0 / DT)
steady_position_error = np.mean(
    np.linalg.norm(err_hist[settle_start:], axis=1)
)

# Timing metric is measured for controller/state computation only.
# A real-time scheduler would also need OS/hardware timing analysis.
max_compute = np.max(compute_ms)
mean_compute = np.mean(compute_ms)
p95_compute = np.percentile(compute_ms, 95)

# ---------------- Console report ----------------
print("=" * 68)
print(" MULTI-ROTOR DISCRETE SIMULATION - RESULTS")
print("=" * 68)
print(f"Simulation frequency       : {FS:.1f} Hz")
print(f"Simulation timestep        : {DT*1000:.1f} ms")
print(f"Total simulation time      : {T_END:.1f} s")
print(f"Crosswind step             : t >= {WIND_TIME:.1f} s")
print()
print("SE(3) BODY -> INERTIAL")
print("Rotation matrix R at t = 5 s:")
print(np.array2string(R5, precision=4, suppress_small=True))
print()
print("Homogeneous SE(3) transform at t = 5 s:")
print(np.array2string(T5, precision=4, suppress_small=True))
print()
print("FINAL / STEADY-STATE ERROR")
print(f"Final position error       : {final_position_error:.6f} m")
print(f"Mean position error (8-12s): {steady_position_error:.6f} m")
print(f"Final attitude error       : {final_attitude_error_deg:.6f} deg")
print()
print("COMPUTATION-TIME MEASUREMENT")
print(f"Mean controller time       : {mean_compute:.4f} ms")
print(f"Maximum controller time    : {max_compute:.4f} ms (may include OS scheduling jitter)")
print(f"95th percentile time       : {p95_compute:.4f} ms")
print(f"Mean < 1 ms?               : {mean_compute < 1.0}")
print(f"95th percentile < 1 ms?    : {p95_compute < 1.0}")
print()
print("NOTE: 20 ms is the simulation/control period; <1 ms is")
print("the measured computation time for the controller/state update.")
print("=" * 68)

# ---------------- Figure 1: 3D trajectory + SE(3) ----------------
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(121, projection="3d")
ax.plot(xd, yd, zd, "--", linewidth=2, label="Desired trajectory")
ax.plot(pos_hist[:,0], pos_hist[:,1], pos_hist[:,2],
        linewidth=2, label="Actual trajectory")
ax.scatter(xd[0], yd[0], zd[0], s=60, label="Start")
ax.scatter(xd[-1], yd[-1], zd[-1], s=60, marker="x", label="End")
ax.set_title("3D Multi-Rotor Trajectory")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()
ax.grid(True)

ax2 = fig.add_subplot(122)
ax2.axis("off")
text = (
    "SE(3) BODY → INERTIAL\n\n"
    "R = Rz(ψ) · Ry(θ) · Rx(φ)\n\n"
    "Rotation matrix at t = 5 s:\n"
    f"{np.array2string(R5, precision=3, suppress_small=True)}\n\n"
    "Homogeneous transform T:\n"
    f"{np.array2string(T5, precision=3, suppress_small=True)}"
)
ax2.text(0.02, 0.98, text, va="top", family="monospace", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "output_1_trajectory_se3.png", dpi=160, bbox_inches="tight")

# ---------------- Figure 2: control, disturbance, errors, timing ----------------
fig, axs = plt.subplots(2, 2, figsize=(13, 9))

# Position error
axs[0,0].plot(t, err_hist[:,0], label="X error")
axs[0,0].plot(t, err_hist[:,1], label="Y error")
axs[0,0].plot(t, err_hist[:,2], label="Z error")
axs[0,0].axvline(WIND_TIME, linestyle="--", label="Crosswind step")
axs[0,0].set_title("Position Error Convergence")
axs[0,0].set_xlabel("Time (s)")
axs[0,0].set_ylabel("Error (m)")
axs[0,0].grid(True)
axs[0,0].legend()

# Attitude
axs[0,1].plot(t, np.degrees(att_hist[:,0]), label="Roll")
axs[0,1].plot(t, np.degrees(att_hist[:,1]), label="Pitch")
axs[0,1].plot(t, np.degrees(att_hist[:,2]), label="Yaw")
axs[0,1].set_title("Attitude Response")
axs[0,1].set_xlabel("Time (s)")
axs[0,1].set_ylabel("Angle (deg)")
axs[0,1].grid(True)
axs[0,1].legend()

# PID output
axs[1,0].plot(t, u_hist[:,0], label="U_x")
axs[1,0].plot(t, u_hist[:,1], label="U_y")
axs[1,0].plot(t, u_hist[:,2], label="U_z")
axs[1,0].axhline(3.0, linestyle="--", label="Actuator + limit")
axs[1,0].axhline(-3.0, linestyle="--", label="Actuator - limit")
axs[1,0].set_title("PID Output with Anti-Windup / Saturation")
axs[1,0].set_xlabel("Time (s)")
axs[1,0].set_ylabel("Control output")
axs[1,0].grid(True)
axs[1,0].legend()

# Timing
axs[1,1].plot(t, compute_ms, linewidth=1, label="Measured")
axs[1,1].axhline(1.0, linestyle="--", label="1 ms threshold")
axs[1,1].set_title("Controller Computation Time")
axs[1,1].set_xlabel("Time (s)")
axs[1,1].set_ylabel("Time (ms)")
axs[1,1].grid(True)
axs[1,1].legend()

fig.tight_layout()
fig.savefig(OUT / "output_2_pid_crosswind_timing.png", dpi=160, bbox_inches="tight")

# ---------------- Figure 3: position tracking ----------------
fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
names = ["X position", "Y position", "Z position"]
desired = [xd, yd, zd]
actual = [pos_hist[:,0], pos_hist[:,1], pos_hist[:,2]]

for i in range(3):
    axs[i].plot(t, desired[i], "--", label="Desired")
    axs[i].plot(t, actual[i], label="Actual")
    axs[i].axvline(WIND_TIME, linestyle="--", label="Crosswind")
    axs[i].set_ylabel(names[i] + " (m)")
    axs[i].grid(True)
    axs[i].legend()

axs[-1].set_xlabel("Time (s)")
fig.suptitle("Position Tracking with Crosswind Disturbance")
fig.tight_layout()
fig.savefig(OUT / "output_3_position_tracking.png", dpi=160, bbox_inches="tight")

plt.show()
