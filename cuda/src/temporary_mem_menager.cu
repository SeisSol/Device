#include <assert.h>

#include "device_utils.h"
#include "temporary_mem_menager.h"

// default size for the tmp mem. = 1GB
#define MAX_DEFAULT_MEMORY 1024*1024*1024


DeviceTemporaryMemoryMenager::~DeviceTemporaryMemoryMenager() {
    device_free(m_base_ptr);
}
        
DeviceTemporaryMemoryMenager& DeviceTemporaryMemoryMenager::get_instance() {
    static DeviceTemporaryMemoryMenager instance;
    return instance;
}

byte* DeviceTemporaryMemoryMenager::get_mem(unsigned amount) {
    assert((m_counter < m_max_mem) && "run of the device temporary mem.");
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
                                                               m_counter(0),
                                                               m_max_mem(MAX_DEFAULT_MEMORY) {

    m_base_ptr = (byte*)device_malloc(m_max_mem);
}
