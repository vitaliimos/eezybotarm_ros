#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace eezybotarm_hardware
{

class EezyBotArmHardware : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(EezyBotArmHardware)

  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  struct ServoConfig
  {
    int channel = -1;
    double rad_min = 0.0;
    double rad_max = 0.0;
    double us_min = 500.0;
    double us_max = 2500.0;
  };

  std::string i2c_bus_ = "/dev/i2c-7";
  int i2c_address_ = 0x60;
  double pwm_frequency_ = 50.0;
  int i2c_fd_ = -1;

  int joint_base_idx_ = -1;
  int joint_shoulder_idx_ = -1;
  int joint_elbow_idx_ = -1;
  int joint_gripper_idx_ = -1;

  std::vector<double> hw_commands_position_;
  std::vector<double> hw_states_position_;

  std::vector<double> gear_ratio_;
  std::vector<bool> invert_direction_;
  std::vector<ServoConfig> servo_config_;

  std::array<std::uint16_t, 16> pwm_on_ticks_{};
  std::array<std::uint16_t, 16> pwm_off_ticks_{};

  bool open_i2c();
  void close_i2c();
  bool init_pca9685();
  bool write_pca9685_all_channels();

  double joint_to_actuator(std::size_t joint_idx, double joint_value) const;
  double rad_to_pulse_us(std::size_t joint_idx, double actuator_rad) const;
};

}  // namespace eezybotarm_hardware
