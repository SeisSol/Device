#include <assert.h>

#include "device_utils.h"
#include "scratch_mem_system.h"

#define SCRATCH_MEM 1024*1024*128


DeviceScratchMem::~DeviceScratchMem() {
            device_free(m_ptr);
}
        
DeviceScratchMem& DeviceScratchMem::get_instance() {
    static DeviceScratchMem instance;
    return instance;
}

real* DeviceScratchMem::get_mem(unsigned amount) {
    assert((m_counter < m_max_mem) && "run of scratch mem.");
    real *return_value = m_ptr + m_counter;
    m_counter += amount;
    m_stack.push(amount);
    return return_value;
}

void DeviceScratchMem::free() {
    m_counter -= m_stack.top();
    m_stack.pop();
}

DeviceScratchMem::DeviceScratchMem() : m_ptr(nullptr),
                                       m_counter(0),
                                       m_max_mem(SCRATCH_MEM) {

    m_ptr = (real*)device_malloc(m_max_mem * sizeof(real));    
}
