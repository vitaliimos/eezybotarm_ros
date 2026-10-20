#pragma once

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

  // channel must be in [0, kNumChannels).
  bool set_pulse_us(int channel, double pulse_us);
  bool all_off();

private:
  bool write_reg(std::uint8_t reg, std::uint8_t value);

  int fd_ = -1;
  double us_per_tick_ = 0.0;
};

}  // namespace eezybotarm_hardware
