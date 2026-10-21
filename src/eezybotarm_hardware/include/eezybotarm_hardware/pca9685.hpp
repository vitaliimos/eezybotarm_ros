#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace eezybotarm_hardware
{

class Pca9685
{
public:
  static constexpr int kNumChannels = 16;

  Pca9685() = default;
  ~Pca9685();
  Pca9685(const Pca9685 &) = delete;
  Pca9685 & operator=(const Pca9685 &) = delete;

  // Opens the bus and configures the chip with all outputs off.
  // Throws std::runtime_error on failure.
  void open(const std::string & bus, int address, double pwm_frequency_hz);
  void close();

  // Stages the pulse width for channel in [0, kNumChannels); call flush() to send.
  void set_pulse_us(int channel, double pulse_us);
  bool flush();
  bool all_off();

private:
  bool write_reg(std::uint8_t reg, std::uint8_t value);

  int fd_ = -1;
  double us_per_tick_ = 0.0;
  std::array<std::uint16_t, kNumChannels> ticks_{};
  int max_channel_ = -1;
  bool dirty_ = false;
};

}  // namespace eezybotarm_hardware
