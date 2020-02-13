#include <assert.h>
#include <iostream>
#include <string>

#include "device_utils.h"
#include "temporary_mem_menager.h"

// default size for the tmp mem. = 1GB
#define MAX_DEFAULT_MEMORY 1


DeviceTemporaryMemoryMenager::~DeviceTemporaryMemoryMenager() {
    device_free(m_base_ptr);
}
        
DeviceTemporaryMemoryMenager& DeviceTemporaryMemoryMenager::get_instance() {
    static DeviceTemporaryMemoryMenager instance;
    return instance;
}

byte* DeviceTemporaryMemoryMenager::get_mem(unsigned amount) {
    assert((m_counter < m_max_mem) && "run out of the device temporary mem.");
    byte *next_ptr = &m_base_ptr[m_counter];
    m_counter += amount;
    m_stack.push(amount);
    return next_ptr;
}

void DeviceTemporaryMemoryMenager::free() {
    m_counter -= m_stack.top();
    m_stack.pop();
}

DeviceTemporaryMemoryMenager::DeviceTemporaryMemoryMenager() : m_base_ptr(nullptr),
                                                               m_counter(0) {
  long long Factor = 1024 * 1024 * 1024;  //!< bytes in 1 GB
  m_max_mem = Factor * MAX_DEFAULT_MEMORY; //!< default mem. size

  // TODO:(RAVIL) the amount of the temp. must be determined from the analysis done by a source code generator
  try {
    char *value_str = std::getenv("DEVICE_STACK_TEMP_MEM_SIZE");
    if (!value_str) {
      std::cout << "DEVICE::INFO: env. variable \"DEVICE_STACK_TEMP_MEM_SIZE\" has not been set. "
                << "The default amount of the device memory (" << MAX_DEFAULT_MEMORY << " GB) "
                << "is going to be used to store temp. variables during execution of compute-kernels\n";
    }
    else {
      double RequestedTempMem = std::stod(std::string(value_str));
      m_max_mem = Factor * RequestedTempMem;
      std::cout << "DEVICE::INFO: env. variable \"DEVICE_STACK_TEMP_MEM_SIZE\" has been detected. "
                << RequestedTempMem << "GB of the device memory is going to be used "
                << "to store temp. variables during execution of compute-kernels\n";
    }
  }
  catch (const std::invalid_argument &err) {
    std::cout << "DEVICE::ERROR: " << err.what() << ". File: " << __FILE__ << ", line: " << __LINE__ << "\n";
    throw err;
  }
  catch (const std::out_of_range& err) {
    std::cout << "DEVICE::ERROR: " << err.what() << ". File: " << __FILE__ << ", line: " << __LINE__ << "\n";
    throw err;
  }

  m_base_ptr = (byte*)device_malloc(m_max_mem);
}
