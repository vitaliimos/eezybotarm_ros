#include "eezybotarm_hardware/eezybotarm_hardware.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace eezybotarm_hardware
{

namespace
{
using CallbackReturn = hardware_interface::CallbackReturn;
using Params = std::unordered_map<std::string, std::string>;

constexpr double kPulseSafetyMinUs = 500.0;
constexpr double kPulseSafetyMaxUs = 2500.0;

// Consecutive failed write cycles tolerated before reporting an error (0.5 s at 50 Hz).
constexpr int kMaxWriteFailures = 25;

const std::string & get_string(const Params & params, const std::string & key)
{
  const auto it = params.find(key);
  if (it == params.end()) {
    throw std::runtime_error("missing parameter '" + key + "'");
  }
  return it->second;
}

double get_double(const Params & params, const std::string & key)
{
  const std::string & value = get_string(params, key);
  try {
    return std::stod(value);
  } catch (const std::exception &) {
    throw std::runtime_error("invalid value '" + value + "' for '" + key + "'");
  }
}

int get_int(const Params & params, const std::string & key)
{
  const std::string & value = get_string(params, key);
  try {
    return std::stoi(value, nullptr, 0);  // base 0 accepts "0x60"
  } catch (const std::exception &) {
    throw std::runtime_error("invalid value '" + value + "' for '" + key + "'");
  }
}

EezyBotArmHardware::Servo parse_servo(const Params & params)
{
  EezyBotArmHardware::Servo s;
  s.channel = get_int(params, "pca_channel");
  s.rad_min = get_double(params, "rad_min");
  s.rad_max = get_double(params, "rad_max");
  s.us_min = get_double(params, "us_min");
  s.us_max = get_double(params, "us_max");
  if (params.count("gear_ratio")) {
    s.gear_ratio = get_double(params, "gear_ratio");
  }
  if (params.count("invert_direction")) {
    s.invert = params.at("invert_direction") == "true";
  }

  if (s.channel < 0 || s.channel >= Pca9685::kNumChannels) {
    throw std::runtime_error("pca_channel out of range [0, 15]");
  }
  if (s.rad_max <= s.rad_min) {
    throw std::runtime_error("rad_max must be greater than rad_min");
  }
  if (s.us_max <= s.us_min) {
    throw std::runtime_error("us_max must be greater than us_min");
  }
  if (s.gear_ratio == 0.0) {
    throw std::runtime_error("gear_ratio must not be 0");
  }
  return s;
}

double pulse_us(const EezyBotArmHardware::Servo & s, double joint_rad)
{
  const double actuator_rad = joint_rad * s.gear_ratio * (s.invert ? -1.0 : 1.0);
  const double t = std::clamp((actuator_rad - s.rad_min) / (s.rad_max - s.rad_min), 0.0, 1.0);
  return std::clamp(s.us_min + t * (s.us_max - s.us_min), kPulseSafetyMinUs, kPulseSafetyMaxUs);
}
}  // namespace

CallbackReturn EezyBotArmHardware::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  servos_.clear();
  hw_states_.assign(info_.joints.size(), 0.0);

  std::string context = "hardware";
  try {
    i2c_bus_ = get_string(info_.hardware_parameters, "i2c_bus");
    i2c_address_ = get_int(info_.hardware_parameters, "i2c_address");
    pwm_frequency_ = get_double(info_.hardware_parameters, "pwm_frequency");
    if (pwm_frequency_ <= 0.0 || pwm_frequency_ > 1526.0) {
      throw std::runtime_error("pwm_frequency out of range (0, 1526]");
    }

    for (std::size_t i = 0; i < info_.joints.size(); i++) {
      const auto & joint = info_.joints[i];
      context = "joint '" + joint.name + "'";
      servos_.push_back(parse_servo(joint.parameters));

      for (const auto & state_if : joint.state_interfaces) {
        if (state_if.name == hardware_interface::HW_IF_POSITION &&
          !state_if.initial_value.empty())
        {
          hw_states_[i] = std::stod(state_if.initial_value);
        }
      }
    }
  } catch (const std::exception & e) {
    RCLCPP_FATAL(get_logger(), "Invalid config for %s: %s", context.c_str(), e.what());
    return CallbackReturn::ERROR;
  }

  hw_commands_ = hw_states_;
  return CallbackReturn::SUCCESS;
}

CallbackReturn EezyBotArmHardware::on_configure(const rclcpp_lifecycle::State &)
{
  try {
    pca_.open(i2c_bus_, i2c_address_, pwm_frequency_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Failed to open PCA9685 at 0x%02X: %s", i2c_address_, e.what());
    return CallbackReturn::ERROR;
  }

  RCLCPP_INFO(get_logger(), "Connected to PCA9685 at 0x%02X on %s (%.1f Hz)",
              i2c_address_, i2c_bus_.c_str(), pwm_frequency_);
  return CallbackReturn::SUCCESS;
}

CallbackReturn EezyBotArmHardware::on_activate(const rclcpp_lifecycle::State &)
{
  hw_commands_ = hw_states_;
  write_failures_ = 0;
  return CallbackReturn::SUCCESS;
}

CallbackReturn EezyBotArmHardware::on_deactivate(const rclcpp_lifecycle::State &)
{
  if (!pca_.all_off()) {
    RCLCPP_WARN(get_logger(), "Failed to turn servos off: %s", std::strerror(errno));
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn EezyBotArmHardware::on_shutdown(const rclcpp_lifecycle::State &)
{
  pca_.all_off();
  pca_.close();
  return CallbackReturn::SUCCESS;
}

// Returning SUCCESS moves the component to UNCONFIGURED, so it can be
// configured and activated again without restarting controller_manager.
CallbackReturn EezyBotArmHardware::on_error(const rclcpp_lifecycle::State &)
{
  pca_.all_off();
  pca_.close();
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> EezyBotArmHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> interfaces;
  for (std::size_t i = 0; i < info_.joints.size(); i++) {
    interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_states_[i]);
  }
  return interfaces;
}

std::vector<hardware_interface::CommandInterface> EezyBotArmHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> interfaces;
  for (std::size_t i = 0; i < info_.joints.size(); i++) {
    interfaces.emplace_back(
      info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_[i]);
  }
  return interfaces;
}

// Open loop: servos report no position, so the state is the last command.
hardware_interface::return_type EezyBotArmHardware::read(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  hw_states_ = hw_commands_;
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type EezyBotArmHardware::write(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  bool ok = true;
  for (std::size_t i = 0; i < servos_.size(); i++) {
    ok &= pca_.set_pulse_us(servos_[i].channel, pulse_us(servos_[i], hw_commands_[i]));
  }

  if (ok) {
    write_failures_ = 0;
    return hardware_interface::return_type::OK;
  }

  if (++write_failures_ >= kMaxWriteFailures) {
    RCLCPP_ERROR(get_logger(), "PCA9685 write failed %d times in a row: %s",
                 write_failures_, std::strerror(errno));
    return hardware_interface::return_type::ERROR;
  }

  RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "PCA9685 write failed: %s",
                       std::strerror(errno));
  return hardware_interface::return_type::OK;
}

}  // namespace eezybotarm_hardware

PLUGINLIB_EXPORT_CLASS(eezybotarm_hardware::EezyBotArmHardware,
                       hardware_interface::SystemInterface)
