# MotorState.msg
# ROS2 message for publishing motor state according to CANopen object dictionary
# Timestamp of the state (optional, in nanoseconds since epoch)
std_msgs/Header header

# Node ID of the motor (CANopen node ID)
uint8 com_status
uint16[] statusword
float32[] actual_position
float32[] actual_velocity
float32[] actual_torque
uint16[] error_code