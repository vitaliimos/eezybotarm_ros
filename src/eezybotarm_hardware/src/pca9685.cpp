#include "eezybotarm_hardware/pca9685.hpp"

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace eezybotarm_hardware
{

namespace
{
constexpr std::uint8_t kMode1 = 0x00;
constexpr std::uint8_t kMode2 = 0x01;
constexpr std::uint8_t kLed0OnL = 0x06;
constexpr std::uint8_t kAllLedOffH = 0xFD;
constexpr std::uint8_t kPrescale = 0xFE;

constexpr std::uint8_t kMode1Sleep = 0x10;
constexpr std::uint8_t kMode1AutoInc = 0x20;
constexpr std::uint8_t kMode2OutDrv = 0x04;
constexpr std::uint8_t kFullOff = 0x10;

constexpr double kOscillatorHz = 25e6;
constexpr double kResolution = 4096.0;
}  // namespace

Pca9685::~Pca9685()
{
  close();
}

void Pca9685::open(const std::string & bus, int address, double pwm_frequency_hz)
{
  close();

  fd_ = ::open(bus.c_str(), O_RDWR);
  if (fd_ < 0 || ioctl(fd_, I2C_SLAVE, address) < 0) {
    const std::string err = std::strerror(errno);
    close();
    throw std::runtime_error(bus + ": " + err);
  }

  const auto prescale = static_cast<std::uint8_t>(
    std::clamp(std::round(kOscillatorHz / (kResolution * pwm_frequency_hz)) - 1.0, 3.0, 255.0));
  us_per_tick_ = 1e6 * (static_cast<double>(prescale) + 1.0) / kOscillatorHz;

  // Prescale can only be written while the oscillator is asleep.
  bool ok = write_reg(kMode1, kMode1Sleep | kMode1AutoInc) &&
    write_reg(kPrescale, prescale) &&
    write_reg(kMode1, kMode1AutoInc);
  usleep(500);  // oscillator start-up
  ok = ok && write_reg(kMode2, kMode2OutDrv) && all_off();

  if (!ok) {
    const std::string err = std::strerror(errno);
    close();
    throw std::runtime_error("PCA9685 init failed: " + err);
  }
}

void Pca9685::close()
{
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

void Pca9685::set_pulse_us(int channel, double pulse_us)
{
  if (!std::isfinite(pulse_us) || channel < 0 || channel >= kNumChannels) {
    return;
  }
  const auto ticks = static_cast<std::uint16_t>(
    std::clamp(std::lround(pulse_us / us_per_tick_), 0L, 4095L));
  max_channel_ = std::max(max_channel_, channel);
  if (ticks != ticks_[channel]) {
    ticks_[channel] = ticks;
    dirty_ = true;
  }
}

bool Pca9685::flush()
{
  if (!dirty_ || max_channel_ < 0) {
    return true;
  }

  std::uint8_t buf[1 + 4 * kNumChannels];
  buf[0] = kLed0OnL;
  for (int ch = 0; ch <= max_channel_; ch++) {
    const std::uint16_t t = ticks_[ch];
    std::uint8_t * p = &buf[1 + 4 * ch];
    p[0] = 0;
    p[1] = 0;
    p[2] = static_cast<std::uint8_t>(t & 0xFF);
    p[3] = static_cast<std::uint8_t>(t >> 8);
  }

  const auto len = static_cast<ssize_t>(1 + 4 * (max_channel_ + 1));
  if (::write(fd_, buf, len) != len) {
    return false;
  }
  dirty_ = false;
  return true;
}

bool Pca9685::all_off()
{
  ticks_.fill(static_cast<std::uint16_t>(kFullOff) << 8);
  max_channel_ = -1;
  dirty_ = false;
  return write_reg(kAllLedOffH, kFullOff);
}

bool Pca9685::write_reg(std::uint8_t reg, std::uint8_t value)
{
  const std::uint8_t buf[2] = {reg, value};
  return ::write(fd_, buf, sizeof(buf)) == static_cast<ssize_t>(sizeof(buf));
}

}  // namespace eezybotarm_hardware
