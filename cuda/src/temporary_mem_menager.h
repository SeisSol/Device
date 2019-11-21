
#ifndef TEMPORARY_MEMORY_MENAGER_H_
#define TEMPORARY_MEMORY_MENAGER_H_

#include <stack>
#include "common.h"

typedef char byte;

class DeviceTemporaryMemoryMenager {
public:
  ~DeviceTemporaryMemoryMenager();
  static DeviceTemporaryMemoryMenager& get_instance();
  byte* get_mem(unsigned amount);
  void free();

private:
  DeviceTemporaryMemoryMenager();
  byte *m_base_ptr;
  unsigned m_counter;
  unsigned m_max_mem;
  std::stack<unsigned> m_stack{};
};

#endif  // TEMPORARY_MEMORY_MENAGER_H_