"""WBC policy observation/action dimensions (matches controller_loader / lambda_wbc)."""

num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3 + 3 + 1
num_priv = 5 + 1 + 12
# Legs-only policy; arm pose via IK on the robot (matches stage1 / lambda_wbc).
num_actions = 12
history_len = 10

flag_observe_gait_commands = True
if flag_observe_gait_commands:
    num_proprio += 5

num_observations = num_proprio * (history_len + 1) + num_priv
