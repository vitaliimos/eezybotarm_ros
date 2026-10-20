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
  us_per_tick_ = 1e6 / (kResolution * pwm_frequency_hz);

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

bool Pca9685::set_pulse_us(int channel, double pulse_us)
{
  const auto ticks = static_cast<std::uint16_t>(
    std::clamp(std::lround(pulse_us / us_per_tick_), 0L, 4095L));
  const std::uint8_t buf[5] = {
    static_cast<std::uint8_t>(kLed0OnL + 4 * channel), 0, 0,
    static_cast<std::uint8_t>(ticks & 0xFF), static_cast<std::uint8_t>(ticks >> 8)};
  return ::write(fd_, buf, sizeof(buf)) == static_cast<ssize_t>(sizeof(buf));
}

bool Pca9685::all_off()
{
  return write_reg(kAllLedOffH, kFullOff);
}

bool Pca9685::write_reg(std::uint8_t reg, std::uint8_t value)
{
  const std::uint8_t buf[2] = {reg, value};
  return ::write(fd_, buf, sizeof(buf)) == static_cast<ssize_t>(sizeof(buf));
}

}  // namespace eezybotarm_hardware
