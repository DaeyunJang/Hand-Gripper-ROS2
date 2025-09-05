# MotorCommand.msg
# Motor operating target values
# builtin_interfaces/Time stamp
std_msgs/Header header
float32[] target_position
float32[] target_velocity
float32[] target_torque
uint16[] control_word
uint8 op_mode
int32 velocity_offset
int16 torque_offset