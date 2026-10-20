#pragma once

#include <string>
#include <vector>

#include "eezybotarm_hardware/pca9685.hpp"
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

  hardware_interface::CallbackReturn on_shutdown(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::CallbackReturn on_error(
    const rclcpp_lifecycle::State & previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  struct Servo
  {
    int channel = 0;
    double rad_min = 0.0;
    double rad_max = 0.0;
    double us_min = 0.0;
    double us_max = 0.0;
    double gear_ratio = 1.0;
    bool invert = false;
  };

private:
  std::string i2c_bus_;
  int i2c_address_ = 0;
  double pwm_frequency_ = 0.0;

  std::vector<Servo> servos_;
  std::vector<double> hw_commands_;
  std::vector<double> hw_states_;

  Pca9685 pca_;
  int write_failures_ = 0;
};

}  // namespace eezybotarm_hardware
