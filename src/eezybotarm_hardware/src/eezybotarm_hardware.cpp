#include "eezybotarm_hardware/eezybotarm_hardware.hpp"

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <string>

#include <rclcpp/rclcpp.hpp>

namespace eezybotarm_hardware
{

namespace
{
constexpr std::uint8_t PCA9685_REG_MODE1 = 0x00;
constexpr std::uint8_t PCA9685_REG_MODE2 = 0x01;
constexpr std::uint8_t PCA9685_REG_PRESCALE = 0xFE;
constexpr std::uint8_t PCA9685_REG_LED0_BASE = 0x06;

constexpr std::uint8_t PCA9685_MODE1_SLEEP = 0x10;
constexpr std::uint8_t PCA9685_MODE1_AUTOINC = 0x20;
constexpr std::uint8_t PCA9685_MODE1_RESTART = 0x80;
constexpr std::uint8_t PCA9685_MODE2_OUTDRV = 0x04;

constexpr double PCA9685_OSCILLATOR_HZ = 25000000.0;
constexpr double PCA9685_RESOLUTION = 4096.0;

constexpr double PULSE_SAFETY_MIN_US = 500.0;
constexpr double PULSE_SAFETY_MAX_US = 2500.0;

bool i2c_write_reg(int fd, std::uint8_t addr, std::uint8_t reg, std::uint8_t value)
{
  if (ioctl(fd, I2C_SLAVE, addr) < 0) {
    return false;
  }
  std::uint8_t buf[2] = {reg, value};
  return ::write(fd, buf, 2) == 2;
}

bool i2c_read_reg(int fd, std::uint8_t addr, std::uint8_t reg, std::uint8_t & value)
{
  if (ioctl(fd, I2C_SLAVE, addr) < 0) {
    return false;
  }
  if (::write(fd, &reg, 1) != 1) {
    return false;
  }
  return ::read(fd, &value, 1) == 1;
}
}  // namespace

hardware_interface::CallbackReturn EezyBotArmHardware::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (hardware_interface::SystemInterface::on_init(params) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  const auto & hardware_params = info_.hardware_parameters;

  if (hardware_params.find("i2c_bus") != hardware_params.end()) {
    i2c_bus_ = hardware_params.at("i2c_bus");
  }

  if (hardware_params.find("i2c_address") != hardware_params.end()) {
    const std::string & address = hardware_params.at("i2c_address");
    try {
      std::size_t converged = 0;
      const int base = (address.rfind("0x", 0) == 0) ? 16 : 10;
      const int value = std::stoi(address, &converged, base);
      if (converged != address.size() || value < 0 || value > 0x7F) {
        RCLCPP_WARN(get_logger(), "Invalid i2c_address '%s', using default 0x%02X",
                    address.c_str(), i2c_address_);
      } else {
        i2c_address_ = value;
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "Failed to parse i2c_address '%s', using default 0x%02X: %s",
                  address.c_str(), i2c_address_, e.what());
    }
  }

  if (hardware_params.find("pwm_frequency") != hardware_params.end()) {
    const std::string & frequency = hardware_params.at("pwm_frequency");
    try {
      const double value = std::stod(frequency);
      if (value > 0.0 && value <= 1526.0) {
        pwm_frequency_ = value;
      } else {
        RCLCPP_WARN(get_logger(),
                    "pwm_frequency '%s' out of range (0, 1526], using default %.1f Hz",
                    frequency.c_str(), pwm_frequency_);
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "Failed to parse pwm_frequency '%s', using default %.1f Hz: %s",
                  frequency.c_str(), pwm_frequency_, e.what());
    }
  }

  hw_commands_position_.assign(info_.joints.size(), 0.0);
  hw_states_position_.assign(info_.joints.size(), 0.0);
  gear_ratio_.assign(info_.joints.size(), 1.0);
  invert_direction_.assign(info_.joints.size(), false);
  servo_config_.assign(info_.joints.size(), ServoConfig());

  for (std::size_t i = 0; i < info_.joints.size(); i++) {
    const auto & name = info_.joints[i].name;
    const auto & parameters = info_.joints[i].parameters;

    if (name == "waist_joint") {
      joint_base_idx_ = static_cast<int>(i);
    } else if (name == "shoulder_joint") {
      joint_shoulder_idx_ = static_cast<int>(i);
    } else if (name == "elbow_joint") {
      joint_elbow_idx_ = static_cast<int>(i);
    } else if (name == "finger_left_joint") {
      joint_gripper_idx_ = static_cast<int>(i);
    }

    ServoConfig & config = servo_config_[i];
    config.channel = static_cast<int>(i);

    auto param_it = parameters.find("pca_channel");
    if (param_it != parameters.end()) {
      try {
        config.channel = std::stoi(param_it->second);
      } catch (const std::exception & e) {
        RCLCPP_WARN(get_logger(), "Invalid pca_channel '%s' for joint '%s', using %d: %s",
                    param_it->second.c_str(), name.c_str(), config.channel, e.what());
      }
    }
    if (config.channel < 0 || config.channel > 15) {
      RCLCPP_WARN(get_logger(), "pca_channel %d for joint '%s' out of range, disabling",
                  config.channel, name.c_str());
      config.channel = -1;
    }

    auto get_double_param = [&](const std::string & key, double fallback) {
        auto it = parameters.find(key);
        if (it == parameters.end()) {
          return fallback;
        }
        try {
          return std::stod(it->second);
        } catch (const std::exception & e) {
          RCLCPP_WARN(get_logger(), "Invalid %s '%s' for joint '%s', using %f: %s", key.c_str(),
                    it->second.c_str(), name.c_str(), fallback, e.what());
          return fallback;
        }
      };

    config.rad_min = get_double_param("rad_min", -M_PI_2);
    config.rad_max = get_double_param("rad_max", M_PI_2);
    config.us_min = get_double_param("us_min", 500.0);
    config.us_max = get_double_param("us_max", 2500.0);

    if (config.rad_max <= config.rad_min) {
      RCLCPP_WARN(get_logger(), "rad_max <= rad_min for joint '%s'", name.c_str());
      config.channel = -1;
    }
    if (config.us_max <= config.us_min) {
      RCLCPP_WARN(get_logger(), "us_max <= us_min for joint '%s'", name.c_str());
      config.channel = -1;
    }

    auto gear_ratio_it = parameters.find("gear_ratio");
    if (gear_ratio_it != parameters.end()) {
      try {
        gear_ratio_[i] = std::stod(gear_ratio_it->second);
      } catch (const std::exception & e) {
        RCLCPP_WARN(get_logger(), "Invalid gear_ratio '%s' for joint '%s', using 1.0: %s",
                    gear_ratio_it->second.c_str(), name.c_str(), e.what());
        gear_ratio_[i] = 1.0;
      }

      if (gear_ratio_[i] == 0.0) {
        RCLCPP_WARN(get_logger(), "gear_ratio for joint '%s' is 0.0, using 1.0 instead",
                    name.c_str());
        gear_ratio_[i] = 1.0;
      }
    }

    auto invert_direction_it = parameters.find("invert_direction");
    if (invert_direction_it != parameters.end()) {
      const std::string & value = invert_direction_it->second;
      invert_direction_[i] = (value == "true" || value == "1");
    }
  }

  if (joint_base_idx_ == -1 || joint_shoulder_idx_ == -1 ||
    joint_elbow_idx_ == -1 || joint_gripper_idx_ == -1)
  {
    RCLCPP_FATAL(get_logger(),
                 "Not all joints found! Expected: waist_joint, "
                 "shoulder_joint, elbow_joint, finger_left_joint.");
    return CallbackReturn::ERROR;
  }

  RCLCPP_INFO(get_logger(), "Hardware Interface initialized.");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn EezyBotArmHardware::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "Connecting to hardware...");

  if (!open_i2c()) {
    RCLCPP_ERROR(get_logger(), "Failed to open I2C bus %s", i2c_bus_.c_str());
    return CallbackReturn::ERROR;
  }

  if (!init_pca9685()) {
    RCLCPP_ERROR(get_logger(), "Failed to initialize PCA9685 at 0x%02X on %s", i2c_address_,
                 i2c_bus_.c_str());
    close_i2c();
    return CallbackReturn::ERROR;
  }

  RCLCPP_INFO(get_logger(), "Successfully connected to PCA9685 at 0x%02X on %s (PWM %.1f Hz)",
              i2c_address_, i2c_bus_.c_str(), pwm_frequency_);

  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn EezyBotArmHardware::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "Activating (Torque ON)...");

  for (std::size_t i = 0; i < hw_states_position_.size(); i++) {
    hw_commands_position_[i] = hw_states_position_[i];
  }

  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn EezyBotArmHardware::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  RCLCPP_INFO(get_logger(), "Deactivating (Torque OFF)...");

  pwm_on_ticks_.fill(0);
  pwm_off_ticks_.fill(0);
  if (i2c_fd_ >= 0) {
    write_pca9685_all_channels();
  }

  close_i2c();

  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
EezyBotArmHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  for (std::size_t i = 0; i < info_.joints.size(); i++) {
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_states_position_[i]));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
EezyBotArmHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;

  for (std::size_t i = 0; i < info_.joints.size(); i++) {
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_position_[i]));
  }

  return command_interfaces;
}

hardware_interface::return_type
EezyBotArmHardware::read(
  const rclcpp::Time & /*time*/,
  const rclcpp::Duration & /*period*/)
{
  for (std::size_t i = 0; i < hw_states_position_.size(); i++) {
    hw_states_position_[i] = hw_commands_position_[i];
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type
EezyBotArmHardware::write(
  const rclcpp::Time & /*time*/,
  const rclcpp::Duration & /*period*/)
{
  if (i2c_fd_ < 0) {
    return hardware_interface::return_type::OK;
  }

  bool channels_updated = false;
  for (std::size_t i = 0; i < info_.joints.size(); i++) {
    const ServoConfig & config = servo_config_[i];
    if (config.channel < 0 || config.channel > 15) {
      continue;
    }

    const double actuator_rad = joint_to_actuator(i, hw_commands_position_[i]);
    const double pulse_us = rad_to_pulse_us(i, actuator_rad);
    const double ticks_double = pulse_us * (PCA9685_RESOLUTION * pwm_frequency_ / 1000000.0);
    const std::uint16_t ticks =
      static_cast<std::uint16_t>(std::lround(std::clamp(ticks_double, 0.0, 4095.0)));

    pwm_on_ticks_[config.channel] = 0;
    pwm_off_ticks_[config.channel] = ticks;
    channels_updated = true;
  }

  if (!channels_updated) {
    return hardware_interface::return_type::OK;
  }

  if (!write_pca9685_all_channels()) {
    RCLCPP_WARN(get_logger(), "Failed to write PWM to PCA9685");
  }

  return hardware_interface::return_type::OK;
}

bool EezyBotArmHardware::open_i2c()
{
  i2c_fd_ = ::open(i2c_bus_.c_str(), O_RDWR);
  if (i2c_fd_ < 0) {
    RCLCPP_ERROR(get_logger(), "Failed to open I2C bus %s: %s", i2c_bus_.c_str(),
                 std::strerror(errno));
    return false;
  }

  if (ioctl(i2c_fd_, I2C_SLAVE, i2c_address_) < 0) {
    RCLCPP_ERROR(get_logger(), "Failed to set I2C slave address 0x%02X: %s", i2c_address_,
                 std::strerror(errno));
    close(i2c_fd_);
    i2c_fd_ = -1;
    return false;
  }

  return true;
}

void EezyBotArmHardware::close_i2c()
{
  if (i2c_fd_ >= 0) {
    ::close(i2c_fd_);
    i2c_fd_ = -1;
  }
}

bool EezyBotArmHardware::init_pca9685()
{
  std::uint8_t mode1 = 0x01;
  if (!i2c_read_reg(i2c_fd_, i2c_address_, PCA9685_REG_MODE1, mode1)) {
    RCLCPP_ERROR(get_logger(), "Failed to read PCA9685 MODE1 register");
    return false;
  }

  if (!i2c_write_reg(i2c_fd_, i2c_address_, PCA9685_REG_MODE1,
                     static_cast<std::uint8_t>(mode1 | PCA9685_MODE1_SLEEP)))
  {
    RCLCPP_ERROR(get_logger(), "Failed to put PCA9685 to sleep");
    return false;
  }

  const double prescale_double =
    std::round(PCA9685_OSCILLATOR_HZ / (PCA9685_RESOLUTION * pwm_frequency_)) - 1.0;
  const std::uint8_t prescale =
    static_cast<std::uint8_t>(std::clamp(prescale_double, 3.0, 255.0));

  if (!i2c_write_reg(i2c_fd_, i2c_address_, PCA9685_REG_PRESCALE, prescale)) {
    RCLCPP_ERROR(get_logger(), "Failed to set PCA9685 prescale");
    return false;
  }

  usleep(500);

  const std::uint8_t wake = static_cast<std::uint8_t>(
    (mode1 & ~PCA9685_MODE1_SLEEP) | PCA9685_MODE1_AUTOINC | PCA9685_MODE1_RESTART);
  if (!i2c_write_reg(i2c_fd_, i2c_address_, PCA9685_REG_MODE1, wake)) {
    RCLCPP_ERROR(get_logger(), "Failed to wake PCA9685");
    return false;
  }

  if (!i2c_write_reg(i2c_fd_, i2c_address_, PCA9685_REG_MODE2, PCA9685_MODE2_OUTDRV)) {
    RCLCPP_ERROR(get_logger(), "Failed to set PCA9685 MODE2");
    return false;
  }

  pwm_on_ticks_.fill(0);
  pwm_off_ticks_.fill(0);
  if (!write_pca9685_all_channels()) {
    RCLCPP_ERROR(get_logger(), "Failed to zero PCA9685 outputs");
    return false;
  }

  return true;
}

bool EezyBotArmHardware::write_pca9685_all_channels()
{
  if (i2c_fd_ < 0) {
    return false;
  }

  if (ioctl(i2c_fd_, I2C_SLAVE, i2c_address_) < 0) {
    return false;
  }

  std::array<std::uint8_t, 1 + 16 * 4> buffer{};
  buffer[0] = PCA9685_REG_LED0_BASE;
  for (std::size_t channel = 0; channel < 16; channel++) {
    const std::size_t base = 1 + channel * 4;
    buffer[base] = static_cast<std::uint8_t>(pwm_on_ticks_[channel] & 0x00FF);
    buffer[base + 1] = static_cast<std::uint8_t>((pwm_on_ticks_[channel] >> 8) & 0x0F);
    buffer[base + 2] = static_cast<std::uint8_t>(pwm_off_ticks_[channel] & 0x00FF);
    buffer[base + 3] = static_cast<std::uint8_t>((pwm_off_ticks_[channel] >> 8) & 0x0F);
  }

  return ::write(i2c_fd_, buffer.data(), buffer.size()) ==
         static_cast<ssize_t>(buffer.size());
}

double EezyBotArmHardware::joint_to_actuator(
  std::size_t joint_idx,
  double joint_value) const
{
  double value = joint_value * gear_ratio_[joint_idx];
  if (invert_direction_[joint_idx]) {
    value = -value;
  }
  return value;
}

double EezyBotArmHardware::rad_to_pulse_us(
  std::size_t joint_idx,
  double actuator_rad) const
{
  const ServoConfig & config = servo_config_[joint_idx];
  const double span = config.rad_max - config.rad_min;
  if (span <= 0.0) {
    return config.us_min;
  }

  const double t = std::clamp((actuator_rad - config.rad_min) / span, 0.0, 1.0);
  const double us = config.us_min + t * (config.us_max - config.us_min);
  return std::clamp(us, PULSE_SAFETY_MIN_US, PULSE_SAFETY_MAX_US);
}

}  // namespace eezybotarm_hardware

PLUGINLIB_EXPORT_CLASS(eezybotarm_hardware::EezyBotArmHardware,
                       hardware_interface::SystemInterface)
