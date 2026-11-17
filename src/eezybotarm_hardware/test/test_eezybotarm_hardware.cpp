#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "eezybotarm_hardware/eezybotarm_hardware.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/types/hardware_component_params.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace
{

class TempPwmChip
{
public:
  explicit TempPwmChip(
    int num_channels = 4,
    const std::filesystem::path & parent = {},
    const std::string & dir_name = {})
  {
    const auto base = parent.empty() ? std::filesystem::temp_directory_path() : parent;
    const auto name = dir_name.empty() ?
      ("eezybotarm_pwmchip_test_" + std::to_string(::getpid()) + "_" + std::to_string(++counter_)) :
      dir_name;
    path_ = base / name;
    std::filesystem::create_directories(path_);
    touch(path_ / "export");
    for (int ch = 0; ch < num_channels; ++ch) {
      const auto ch_dir = path_ / ("pwm" + std::to_string(ch));
      std::filesystem::create_directories(ch_dir);
      touch(ch_dir / "period");
      touch(ch_dir / "duty_cycle");
      touch(ch_dir / "enable");
    }
  }

  ~TempPwmChip()
  {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  const std::filesystem::path & path() const {return path_;}

  std::string read_first_line(int ch, const std::string & attr) const
  {
    std::ifstream in(path_ / ("pwm" + std::to_string(ch)) / attr);
    std::string line;
    std::getline(in, line);
    return line;
  }

  void write_attr(int ch, const std::string & attr, const std::string & value) const
  {
    std::ofstream out(path_ / ("pwm" + std::to_string(ch)) / attr);
    out << value;
  }

private:
  static void touch(const std::filesystem::path & file)
  {
    std::ofstream out(file);
  }

  static inline int counter_ = 0;
  std::filesystem::path path_;
};

hardware_interface::ComponentInfo make_joint(
  const std::string & name,
  int pwm_channel,
  const std::string & initial_value,
  const std::string & gear_ratio = "",
  bool invert = false)
{
  hardware_interface::ComponentInfo joint;
  joint.name = name;
  joint.type = "joint";

  hardware_interface::InterfaceInfo cmd;
  cmd.name = hardware_interface::HW_IF_POSITION;
  joint.command_interfaces.push_back(cmd);

  hardware_interface::InterfaceInfo state;
  state.name = hardware_interface::HW_IF_POSITION;
  state.initial_value = initial_value;
  joint.state_interfaces.push_back(state);

  joint.parameters["pwm_channel"] = std::to_string(pwm_channel);
  joint.parameters["rad_min"] = "-1.5708";
  joint.parameters["rad_max"] = "1.5708";
  joint.parameters["us_min"] = "600";
  joint.parameters["us_max"] = "2400";
  if (!gear_ratio.empty()) {
    joint.parameters["gear_ratio"] = gear_ratio;
  }
  if (invert) {
    joint.parameters["invert_direction"] = "true";
  }
  return joint;
}

hardware_interface::HardwareInfo make_hardware_info(
  const std::string & pwm_chip, const std::string & pwm_frequency = "50")
{
  hardware_interface::HardwareInfo info;
  info.name = "EezyBotArmSystem";
  info.type = "system";
  info.hardware_parameters["pwm_chip"] = pwm_chip;
  info.hardware_parameters["pwm_frequency"] = pwm_frequency;
  info.joints.push_back(make_joint("waist_joint", 0, "0.0", "2.0", true));
  info.joints.push_back(make_joint("shoulder_joint", 1, "-0.45"));
  info.joints.push_back(make_joint("elbow_joint", 2, "-0.25", "", true));
  info.joints.push_back(make_joint("finger_left_joint", 3, "0.0"));
  return info;
}

hardware_interface::CallbackReturn init_hw(
  eezybotarm_hardware::EezyBotArmHardware & hw, const hardware_interface::HardwareInfo & info)
{
  hardware_interface::HardwareComponentParams params;
  params.hardware_info = info;
  params.logger = rclcpp::get_logger("test_eezybotarm_hardware");
  params.clock = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
  return hw.init(params);
}

long expected_duty_ns(double joint_rad, double gear_ratio, bool invert)
{
  const double actuator_rad = joint_rad * gear_ratio * (invert ? -1.0 : 1.0);
  const double t = std::clamp((actuator_rad - (-1.5708)) / (1.5708 - (-1.5708)), 0.0, 1.0);
  const double us = std::clamp(600.0 + t * (2400.0 - 600.0), 500.0, 2500.0);
  return std::lround(us * 1000.0);
}

TEST(EezyBotArmHardwareTest, LifecycleAndPulseConversion)
{
  TempPwmChip chip(4);
  eezybotarm_hardware::EezyBotArmHardware hw;

  ASSERT_EQ(
    init_hw(hw, make_hardware_info(chip.path().string())),
    hardware_interface::CallbackReturn::SUCCESS);

  rclcpp_lifecycle::State dummy_state;
  ASSERT_EQ(hw.on_configure(dummy_state), hardware_interface::CallbackReturn::SUCCESS);

  for (int ch = 0; ch < 4; ++ch) {
    EXPECT_EQ(chip.read_first_line(ch, "period"), "20000000");
  }

  ASSERT_EQ(hw.on_activate(dummy_state), hardware_interface::CallbackReturn::SUCCESS);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  auto state_ifs = hw.export_state_interfaces();
  auto cmd_ifs = hw.export_command_interfaces();
#pragma GCC diagnostic pop
  ASSERT_EQ(state_ifs.size(), 4u);
  ASSERT_EQ(cmd_ifs.size(), 4u);

  const rclcpp::Time t(0, 0, RCL_STEADY_TIME);
  const rclcpp::Duration dt = rclcpp::Duration::from_seconds(0.02);

  ASSERT_EQ(hw.write(t, dt), hardware_interface::return_type::OK);

  EXPECT_EQ(std::stol(chip.read_first_line(0, "duty_cycle")), expected_duty_ns(0.0, 2.0, true));
  EXPECT_EQ(std::stol(chip.read_first_line(1, "duty_cycle")), expected_duty_ns(-0.45, 1.0, false));
  EXPECT_EQ(std::stol(chip.read_first_line(2, "duty_cycle")), expected_duty_ns(-0.25, 1.0, true));
  EXPECT_EQ(std::stol(chip.read_first_line(3, "duty_cycle")), expected_duty_ns(0.0, 1.0, false));
  for (int ch = 0; ch < 4; ++ch) {
    EXPECT_EQ(chip.read_first_line(ch, "enable"), "1");
  }

  ASSERT_TRUE(cmd_ifs[0].set_value(0.5));
  ASSERT_EQ(hw.write(t, dt), hardware_interface::return_type::OK);
  ASSERT_EQ(hw.read(t, dt), hardware_interface::return_type::OK);

  EXPECT_EQ(std::stol(chip.read_first_line(0, "duty_cycle")), expected_duty_ns(0.5, 2.0, true));
  EXPECT_DOUBLE_EQ(state_ifs[0].get_optional<double>().value(), 0.5);

  ASSERT_EQ(hw.on_deactivate(dummy_state), hardware_interface::CallbackReturn::SUCCESS);
  for (int ch = 0; ch < 4; ++ch) {
    EXPECT_EQ(chip.read_first_line(ch, "enable"), "0");
  }

  ASSERT_EQ(hw.on_activate(dummy_state), hardware_interface::CallbackReturn::SUCCESS);
  ASSERT_EQ(hw.write(t, dt), hardware_interface::return_type::OK);
  for (int ch = 0; ch < 4; ++ch) {
    EXPECT_EQ(chip.read_first_line(ch, "enable"), "1");
  }

  ASSERT_EQ(hw.on_shutdown(dummy_state), hardware_interface::CallbackReturn::SUCCESS);
  for (int ch = 0; ch < 4; ++ch) {
    EXPECT_EQ(chip.read_first_line(ch, "enable"), "0");
  }
}

TEST(EezyBotArmHardwareTest, RejectsInvalidConfig)
{
  {
    eezybotarm_hardware::EezyBotArmHardware hw;
    auto info = make_hardware_info("/tmp/unused", "10.0");
    EXPECT_EQ(init_hw(hw, info), hardware_interface::CallbackReturn::ERROR);
  }
  {
    eezybotarm_hardware::EezyBotArmHardware hw;
    auto info = make_hardware_info("/tmp/unused", "nan");
    EXPECT_EQ(init_hw(hw, info), hardware_interface::CallbackReturn::ERROR);
  }
  {
    eezybotarm_hardware::EezyBotArmHardware hw;
    auto info = make_hardware_info("/tmp/unused", "50");
    info.hardware_parameters.erase("pwm_frequency");
    EXPECT_EQ(init_hw(hw, info), hardware_interface::CallbackReturn::ERROR);
  }
  {
    eezybotarm_hardware::EezyBotArmHardware hw;
    auto info = make_hardware_info("/tmp/unused", "500.0");
    EXPECT_EQ(init_hw(hw, info), hardware_interface::CallbackReturn::ERROR);
  }
  {
    eezybotarm_hardware::EezyBotArmHardware hw;
    auto info = make_hardware_info("/tmp/unused", "50");
    info.hardware_parameters.erase("pwm_chip");
    EXPECT_EQ(init_hw(hw, info), hardware_interface::CallbackReturn::ERROR);
  }
  {
    eezybotarm_hardware::EezyBotArmHardware hw;
    auto info = make_hardware_info("", "50");
    EXPECT_EQ(init_hw(hw, info), hardware_interface::CallbackReturn::ERROR);
  }
  {
    eezybotarm_hardware::EezyBotArmHardware hw;
    auto info = make_hardware_info("/tmp/unused", "50");
    info.joints[1].parameters["pwm_channel"] = "0";
    EXPECT_EQ(init_hw(hw, info), hardware_interface::CallbackReturn::ERROR);
  }
  {
    eezybotarm_hardware::EezyBotArmHardware hw;
    auto info = make_hardware_info("/tmp/unused", "50");
    info.joints[1].parameters["pwm_channel"] = "16";
    EXPECT_EQ(init_hw(hw, info), hardware_interface::CallbackReturn::ERROR);
  }
}

TEST(EezyBotArmHardwareTest, ConfigureFailsOnMissingChipDir)
{
  eezybotarm_hardware::EezyBotArmHardware hw;
  ASSERT_EQ(
    init_hw(hw, make_hardware_info("/tmp/nonexistent_pwmchip_dir_12345")),
    hardware_interface::CallbackReturn::SUCCESS);

  rclcpp_lifecycle::State dummy_state;
  EXPECT_EQ(hw.on_configure(dummy_state), hardware_interface::CallbackReturn::ERROR);
}

TEST(EezyBotArmHardwareTest, DisablesUnconfiguredExportedChannelsOnConfigure)
{
  TempPwmChip chip(16);
  chip.write_attr(4, "enable", "1\n");
  chip.write_attr(4, "duty_cycle", "1500000\n");
  chip.write_attr(15, "enable", "1\n");
  chip.write_attr(15, "duty_cycle", "1200000\n");

  eezybotarm_hardware::EezyBotArmHardware hw;
  ASSERT_EQ(
    init_hw(hw, make_hardware_info(chip.path().string())),
    hardware_interface::CallbackReturn::SUCCESS);

  rclcpp_lifecycle::State dummy_state;
  ASSERT_EQ(hw.on_configure(dummy_state), hardware_interface::CallbackReturn::SUCCESS);

  for (int ch = 0; ch < 4; ++ch) {
    EXPECT_EQ(chip.read_first_line(ch, "period"), "20000000");
    EXPECT_EQ(chip.read_first_line(ch, "enable"), "0");
    EXPECT_EQ(chip.read_first_line(ch, "duty_cycle"), "0");
  }

  EXPECT_EQ(chip.read_first_line(4, "enable"), "0");
  EXPECT_EQ(chip.read_first_line(4, "duty_cycle"), "0");
  EXPECT_EQ(chip.read_first_line(15, "enable"), "0");
  EXPECT_EQ(chip.read_first_line(15, "duty_cycle"), "0");
}

TEST(EezyBotArmHardwareTest, ResolvesSinglePwmChipInsideParentDir)
{
  const auto parent = std::filesystem::temp_directory_path() /
    ("eezybotarm_pwm_parent_" + std::to_string(::getpid()));
  std::error_code ec;
  std::filesystem::remove_all(parent, ec);
  std::filesystem::create_directories(parent);

  {
    eezybotarm_hardware::EezyBotArmHardware hw;
    ASSERT_EQ(
      init_hw(hw, make_hardware_info(parent.string())),
      hardware_interface::CallbackReturn::SUCCESS);
    rclcpp_lifecycle::State dummy_state;
    EXPECT_EQ(hw.on_configure(dummy_state), hardware_interface::CallbackReturn::ERROR);
  }

  {
    TempPwmChip chip(4, parent, "pwmchip3");
    eezybotarm_hardware::EezyBotArmHardware hw;
    ASSERT_EQ(
      init_hw(hw, make_hardware_info(parent.string())),
      hardware_interface::CallbackReturn::SUCCESS);
    rclcpp_lifecycle::State dummy_state;
    ASSERT_EQ(hw.on_configure(dummy_state), hardware_interface::CallbackReturn::SUCCESS);
    EXPECT_EQ(chip.read_first_line(0, "period"), "20000000");

    TempPwmChip second_chip(4, parent, "pwmchip4");
    eezybotarm_hardware::EezyBotArmHardware hw_ambiguous;
    ASSERT_EQ(
      init_hw(hw_ambiguous, make_hardware_info(parent.string())),
      hardware_interface::CallbackReturn::SUCCESS);
    EXPECT_EQ(hw_ambiguous.on_configure(dummy_state), hardware_interface::CallbackReturn::ERROR);
  }

  std::filesystem::remove_all(parent, ec);
}

}  // namespace

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int ret = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return ret;
}
