#include "eezybotarm_hardware/eezybotarm_hardware.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace eezybotarm_hardware
{

namespace
{
using CallbackReturn = hardware_interface::CallbackReturn;
using Params = std::unordered_map<std::string, std::string>;

constexpr double kMinPwmFrequencyHz = 24.0;
constexpr double kMaxPwmFrequencyHz = 1526.0;
constexpr double kPulseSafetyMinUs = 500.0;
constexpr double kPulseSafetyMaxUs = 2500.0;

// Consecutive failed write cycles tolerated before reporting an error (0.5s at 50 Hz)
constexpr int kMaxWriteFailures = 25;

const std::string & get_string(const Params & params, const std::string & key)
{
  const auto it = params.find(key);
  if (it == params.end()) {
    throw std::runtime_error("missing parameter '" + key + "'");
  }
  return it->second;
}

double parse_finite_double(const std::string & value, const std::string & key)
{
  try {
    const double result = std::stod(value);
    if (!std::isfinite(result)) {
      throw std::runtime_error("non-finite value");
    }
    return result;
  } catch (const std::exception &) {
    throw std::runtime_error("invalid value '" + value + "' for '" + key + "'");
  }
}

double get_double(const Params & params, const std::string & key)
{
  return parse_finite_double(get_string(params, key), key);
}

int get_int(const Params & params, const std::string & key)
{
  const std::string & value = get_string(params, key);
  try {
    return std::stoi(value, nullptr, 0);
  } catch (const std::exception &) {
    throw std::runtime_error("invalid value '" + value + "' for '" + key + "'");
  }
}

EezyBotArmHardware::Servo parse_servo(const Params & params)
{
  EezyBotArmHardware::Servo s;
  s.channel = get_int(params, "pwm_channel");
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

  if (s.channel < 0 || s.channel >= EezyBotArmHardware::kMaxChannels) {
    throw std::runtime_error("pwm_channel out of range [0, 15]");
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
  if (!std::isfinite(joint_rad)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double actuator_rad =
    joint_rad * s.gear_ratio * (s.invert ? -1.0 : 1.0);
  const double t = std::clamp(
    (actuator_rad - s.rad_min) / (s.rad_max - s.rad_min), 0.0, 1.0);
  return std::clamp(s.us_min + t * (s.us_max - s.us_min), kPulseSafetyMinUs,
                    kPulseSafetyMaxUs);
}

bool write_fd_str(int fd, const char *data, std::size_t len)
{
  if (fd < 0) {
    errno = EBADF;
    return false;
  }
  std::size_t written = 0;
  while (written < len) {
    const ssize_t n = ::pwrite(fd, data + written, len - written,
                               static_cast<off_t>(written));
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    written += static_cast<std::size_t>(n);
  }
  return true;
}

void write_sysfs_file(
  const std::filesystem::path & path,
  const std::string & value)
{
  const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error(path.string() + ": " + std::strerror(errno));
  }
  const bool ok = write_fd_str(fd, value.data(), value.size());
  const int saved_errno = errno;
  ::close(fd);
  if (!ok) {
    throw std::system_error(saved_errno, std::generic_category(), path.string());
  }
}

int open_sysfs_wr(const std::filesystem::path & path)
{
  const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error(path.string() + ": " + std::strerror(errno));
  }
  return fd;
}

std::filesystem::path resolve_pwm_chip(const std::string & pwm_chip)
{
  std::filesystem::path path(pwm_chip);
  if (!std::filesystem::is_directory(path)) {
    throw std::runtime_error("PWM chip directory does not exist: " + pwm_chip);
  }
  if (std::filesystem::exists(path / "export")) {
    return path;
  }

  std::filesystem::path matched;
  int match_count = 0;
  for (const auto & entry : std::filesystem::directory_iterator(path)) {
    if (!entry.is_directory()) {
      continue;
    }
    const std::string name = entry.path().filename().string();
    if (name.rfind("pwmchip", 0) != 0) {
      continue;
    }
    matched = entry.path();
    ++match_count;
  }

  if (match_count == 0) {
    throw std::runtime_error("no pwmchip* found in: " + pwm_chip);
  }
  if (match_count > 1) {
    throw std::runtime_error("multiple pwmchip* directories found in: " +
                             pwm_chip);
  }
  return matched;
}
} // namespace

EezyBotArmHardware::~EezyBotArmHardware()
{
  disable_all();
  close_fds();
}

void EezyBotArmHardware::close_fds()
{
  for (auto & s : servos_) {
    if (s.duty_fd >= 0) {
      ::close(s.duty_fd);
      s.duty_fd = -1;
    }
    if (s.enable_fd >= 0) {
      ::close(s.enable_fd);
      s.enable_fd = -1;
    }
    s.last_duty_ns = -1;
    s.enabled = false;
  }
}

bool EezyBotArmHardware::disable_all()
{
  bool ok = true;
  for (auto & s : servos_) {
    if (s.enable_fd >= 0) {
      if (!write_fd_str(s.enable_fd, "0\n", 2)) {
        ok = false;
      }
      s.enabled = false;
      s.last_duty_ns = -1;
    }
  }
  return ok;
}

CallbackReturn EezyBotArmHardware::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  close_fds();
  servos_.clear();
  hw_states_.assign(info_.joints.size(), 0.0);

  std::string context = "hardware";
  try {
    const double pwm_frequency =
      get_double(info_.hardware_parameters, "pwm_frequency");
    if (pwm_frequency < kMinPwmFrequencyHz ||
      pwm_frequency > kMaxPwmFrequencyHz)
    {
      throw std::runtime_error("pwm_frequency out of range [24, 1526]");
    }
    period_ns_ = std::lround(1e9 / pwm_frequency);
    if (period_ns_ < static_cast<long>(kPulseSafetyMaxUs * 1000.0)) {
      throw std::runtime_error(
          "pwm_frequency too high for max safety pulse width (" +
          std::to_string(static_cast<int>(kPulseSafetyMaxUs)) + " us)");
    }

    pwm_chip_ = get_string(info_.hardware_parameters, "pwm_chip");
    if (pwm_chip_.empty()) {
      throw std::runtime_error("empty parameter 'pwm_chip'");
    }

    std::unordered_set<int> used_channels;
    for (std::size_t i = 0; i < info_.joints.size(); i++) {
      const auto & joint = info_.joints[i];
      context = "joint '" + joint.name + "'";
      const auto servo = parse_servo(joint.parameters);
      if (!used_channels.insert(servo.channel).second) {
        throw std::runtime_error("duplicate pwm_channel " +
                                 std::to_string(servo.channel));
      }
      servos_.push_back(servo);

      for (const auto & state_if : joint.state_interfaces) {
        if (state_if.name == hardware_interface::HW_IF_POSITION &&
          !state_if.initial_value.empty())
        {
          hw_states_[i] =
            parse_finite_double(state_if.initial_value, "initial_value");
        }
      }
    }
  } catch (const std::exception & e) {
    RCLCPP_FATAL(get_logger(), "Invalid config for %s: %s", context.c_str(),
                 e.what());
    return CallbackReturn::ERROR;
  }

  hw_commands_ = hw_states_;
  return CallbackReturn::SUCCESS;
}

CallbackReturn
EezyBotArmHardware::on_configure(const rclcpp_lifecycle::State &)
{
  close_fds();
  try {
    const auto chip_path = resolve_pwm_chip(pwm_chip_);

    // Ensure configured channels are exported, then disable all
    // already-exported individual channels [0, kMaxChannels) before changing
    // the shared period (prescaler).
    for (const auto & s : servos_) {
      const auto ch_dir = chip_path / ("pwm" + std::to_string(s.channel));
      if (!std::filesystem::exists(ch_dir)) {
        write_sysfs_file(chip_path / "export", std::to_string(s.channel) + "\n");
      }
    }

    for (int ch = 0; ch < kMaxChannels; ++ch) {
      const auto ch_dir = chip_path / ("pwm" + std::to_string(ch));
      if (std::filesystem::exists(ch_dir)) {
        write_sysfs_file(ch_dir / "enable", "0\n");
        write_sysfs_file(ch_dir / "duty_cycle", "0\n");
      }
    }

    for (auto & s : servos_) {
      const auto ch_dir = chip_path / ("pwm" + std::to_string(s.channel));
      write_sysfs_file(ch_dir / "period", std::to_string(period_ns_) + "\n");
      s.duty_fd = open_sysfs_wr(ch_dir / "duty_cycle");
      s.enable_fd = open_sysfs_wr(ch_dir / "enable");
      s.last_duty_ns = -1;
      s.enabled = false;
    }

    RCLCPP_INFO(get_logger(), "Configured PWM chip %s (period %ld ns)",
                chip_path.c_str(), period_ns_);
  } catch (const std::system_error & e) {
    close_fds();
    const char *busy_hint =
      (e.code().value() == EBUSY) ?
      " (possible PCA9685 shared period conflict with another enabled channel)" :
      "";
    RCLCPP_ERROR(get_logger(), "Failed to configure PWM chip: %s%s", e.what(),
                 busy_hint);
    return CallbackReturn::ERROR;
  } catch (const std::exception & e) {
    close_fds();
    RCLCPP_ERROR(get_logger(), "Failed to configure PWM chip: %s", e.what());
    return CallbackReturn::ERROR;
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn
EezyBotArmHardware::on_activate(const rclcpp_lifecycle::State &)
{
  hw_commands_ = hw_states_;
  write_failures_ = 0;
  return CallbackReturn::SUCCESS;
}

CallbackReturn
EezyBotArmHardware::on_deactivate(const rclcpp_lifecycle::State &)
{
  if (!disable_all()) {
    RCLCPP_WARN(get_logger(), "Failed to disable PWM channels: %s",
                std::strerror(errno));
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn
EezyBotArmHardware::on_shutdown(const rclcpp_lifecycle::State &)
{
  disable_all();
  close_fds();
  return CallbackReturn::SUCCESS;
}

// Returning SUCCESS moves the component to UNCONFIGURED, so it can be
// configured and activated again without restarting controller_manager.
CallbackReturn EezyBotArmHardware::on_error(const rclcpp_lifecycle::State &)
{
  disable_all();
  close_fds();
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
EezyBotArmHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> interfaces;
  for (std::size_t i = 0; i < info_.joints.size(); i++) {
    interfaces.emplace_back(info_.joints[i].name,
                            hardware_interface::HW_IF_POSITION, &hw_states_[i]);
  }
  return interfaces;
}

std::vector<hardware_interface::CommandInterface>
EezyBotArmHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> interfaces;
  for (std::size_t i = 0; i < info_.joints.size(); i++) {
    interfaces.emplace_back(info_.joints[i].name,
                            hardware_interface::HW_IF_POSITION,
                            &hw_commands_[i]);
  }
  return interfaces;
}

// Open loop: servos report no position, so the state is the last command.
hardware_interface::return_type
EezyBotArmHardware::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  for (std::size_t i = 0; i < hw_commands_.size(); i++) {
    if (std::isfinite(hw_commands_[i])) {
      hw_states_[i] = hw_commands_[i];
    }
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type
EezyBotArmHardware::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  bool ok = true;
  int last_errno = 0;

  for (std::size_t i = 0; i < servos_.size(); i++) {
    if (!std::isfinite(hw_commands_[i])) {
      continue;
    }
    const double us = pulse_us(servos_[i], hw_commands_[i]);
    if (!std::isfinite(us)) {
      continue;
    }
    const long duty_ns = std::lround(us * 1000.0);
    auto & s = servos_[i];
    if (duty_ns != s.last_duty_ns) {
      const std::string val = std::to_string(duty_ns) + "\n";
      if (!write_fd_str(s.duty_fd, val.data(), val.size())) {
        ok = false;
        last_errno = errno;
        continue;
      }
      s.last_duty_ns = duty_ns;
    }
    if (!s.enabled) {
      if (!write_fd_str(s.enable_fd, "1\n", 2)) {
        ok = false;
        last_errno = errno;
        continue;
      }
      s.enabled = true;
    }
  }

  if (ok) {
    write_failures_ = 0;
    return hardware_interface::return_type::OK;
  }

  const char *busy_hint =
    (last_errno == EBUSY) ?
    " (possible PCA9685 shared period conflict with another enabled channel)" :
    "";
  if (++write_failures_ >= kMaxWriteFailures) {
    RCLCPP_ERROR(get_logger(), "PWM sysfs write failed %d times in a row: %s%s",
                 write_failures_, std::strerror(last_errno), busy_hint);
    return hardware_interface::return_type::ERROR;
  }

  RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                       "PWM sysfs write failed: %s%s", std::strerror(last_errno),
                       busy_hint);
  return hardware_interface::return_type::OK;
}

} // namespace eezybotarm_hardware

PLUGINLIB_EXPORT_CLASS(eezybotarm_hardware::EezyBotArmHardware,
                       hardware_interface::SystemInterface)
