
#ifndef SCRATCH_MEM_SYSTEM_H_
#define SCRATCH_MEM_SYSTEM_H_

#include <stack>
#include "common.h"

class DeviceScratchMem {
    public:
        ~DeviceScratchMem();
        static DeviceScratchMem& get_instance();
        real* get_mem(unsigned amount);
        void free();

    private:
        DeviceScratchMem();

        real *m_ptr;
        unsigned m_counter;
        unsigned m_max_mem;
        std::stack<unsigned> m_stack{};    
};

#endif  // SCRATCH_MEM_SYSTEM_H_