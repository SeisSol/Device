// SPDX-FileCopyrightText: 2026 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause

#include "Common.h"
#include "Runtime.h"
#include "utils/logger.h"

#include <amd_comgr/amd_comgr.h>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hsa/amd_hsa_queue.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <string>
#include <unordered_map>

namespace device::hsaqueues {

namespace {

ArgKind argKind(const std::string& kind) {
  static const std::unordered_map<std::string, ArgKind> kinds{
      {"hidden_block_count_x", ArgKind::BlockCountX},
      {"hidden_block_count_y", ArgKind::BlockCountY},
      {"hidden_block_count_z", ArgKind::BlockCountZ},
      {"hidden_group_size_x", ArgKind::GroupSizeX},
      {"hidden_group_size_y", ArgKind::GroupSizeY},
      {"hidden_group_size_z", ArgKind::GroupSizeZ},
      {"hidden_grid_dims", ArgKind::GridDims},
      {"hidden_dynamic_lds_size", ArgKind::DynamicLdsSize},
      {"hidden_queue_ptr", ArgKind::QueuePtr},
      {"hidden_private_base", ArgKind::PrivateBase},
      {"hidden_shared_base", ArgKind::SharedBase},
      {"hidden_remainder_x", ArgKind::Zero},
      {"hidden_remainder_y", ArgKind::Zero},
      {"hidden_remainder_z", ArgKind::Zero},
      {"hidden_global_offset_x", ArgKind::Zero},
      {"hidden_global_offset_y", ArgKind::Zero},
      {"hidden_global_offset_z", ArgKind::Zero},
      {"hidden_none", ArgKind::Zero},
      {"hidden_printf_buffer", ArgKind::Unsupported},
      {"hidden_hostcall_buffer", ArgKind::Unsupported},
      {"hidden_heap_v1", ArgKind::Unsupported},
      {"hidden_multigrid_sync_arg", ArgKind::Unsupported},
      {"hidden_default_queue", ArgKind::Unsupported},
      {"hidden_completion_action", ArgKind::Unsupported},
  };
  if (kind.rfind("hidden_", 0) != 0) {
    return ArgKind::Explicit;
  }
  const auto found = kinds.find(kind);
  if (found == kinds.end()) {
    warnOnce("unknown hidden kernel argument " + kind + " left at 0");
    return ArgKind::Zero;
  }
  return found->second;
}

std::string metadataString(amd_comgr_metadata_node_t node) {
  std::size_t size = 0;
  checkComgr(amd_comgr_get_metadata_string(node, &size, nullptr), "get_metadata_string");
  std::string text(size, '\0');
  checkComgr(amd_comgr_get_metadata_string(node, &size, text.data()), "get_metadata_string");
  while (!text.empty() && text.back() == '\0') {
    text.pop_back();
  }
  return text;
}

bool lookupString(amd_comgr_metadata_node_t map, const char* key, std::string& value) {
  amd_comgr_metadata_node_t node;
  if (amd_comgr_metadata_lookup(map, key, &node) != AMD_COMGR_STATUS_SUCCESS) {
    return false;
  }
  amd_comgr_metadata_kind_t kind{};
  checkComgr(amd_comgr_get_metadata_kind(node, &kind), "get_metadata_kind");
  const bool found = kind == AMD_COMGR_METADATA_KIND_STRING;
  if (found) {
    value = metadataString(node);
  }
  amd_comgr_destroy_metadata(node);
  return found;
}

std::uint32_t lookupNumber(amd_comgr_metadata_node_t map, const char* key, std::uint32_t fallback) {
  std::string value;
  return lookupString(map, key, value) ? static_cast<std::uint32_t>(std::stoul(value)) : fallback;
}

void readMetadata(Runtime& runtime, const void* elf, std::size_t size) {
  amd_comgr_data_t data;
  checkComgr(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data), "create_data");
  checkComgr(amd_comgr_set_data(data, size, static_cast<const char*>(elf)), "set_data");
  amd_comgr_metadata_node_t root;
  if (amd_comgr_get_data_metadata(data, &root) != AMD_COMGR_STATUS_SUCCESS) {
    amd_comgr_release_data(data);
    return;
  }
  amd_comgr_metadata_node_t kernels;
  if (amd_comgr_metadata_lookup(root, "amdhsa.kernels", &kernels) == AMD_COMGR_STATUS_SUCCESS) {
    std::size_t count = 0;
    checkComgr(amd_comgr_get_metadata_list_size(kernels, &count), "get_metadata_list_size");
    for (std::size_t k = 0; k < count; ++k) {
      amd_comgr_metadata_node_t kernel;
      checkComgr(amd_comgr_index_list_metadata(kernels, k, &kernel), "index_list_metadata");
      std::string symbol;
      lookupString(kernel, ".symbol", symbol);
      auto& info = runtime.symbols[symbol];
      if (info.metadata) {
        // (each code object gets read once: another one has a kernel of the same name)
        info.ambiguous = true;
        amd_comgr_destroy_metadata(kernel);
        continue;
      }
      lookupString(kernel, ".name", info.name);
      info.metadata = true;
      info.kernargSize = lookupNumber(kernel, ".kernarg_segment_size", info.kernargSize);
      info.kernargAlign = std::max<std::uint32_t>(
          16, lookupNumber(kernel, ".kernarg_segment_align", info.kernargAlign));
      info.groupSize = lookupNumber(kernel, ".group_segment_fixed_size", info.groupSize);
      info.privateSize = lookupNumber(kernel, ".private_segment_fixed_size", info.privateSize);
      std::string stack;
      info.dynamicStack =
          lookupString(kernel, ".uses_dynamic_stack", stack) && (stack == "1" || stack == "true");
      info.args.clear();
      info.unsupported = false;
      amd_comgr_metadata_node_t args;
      if (amd_comgr_metadata_lookup(kernel, ".args", &args) == AMD_COMGR_STATUS_SUCCESS) {
        std::size_t argCount = 0;
        checkComgr(amd_comgr_get_metadata_list_size(args, &argCount), "get_metadata_list_size");
        for (std::size_t a = 0; a < argCount; ++a) {
          amd_comgr_metadata_node_t arg;
          checkComgr(amd_comgr_index_list_metadata(args, a, &arg), "index_list_metadata");
          std::string kind;
          lookupString(arg, ".value_kind", kind);
          ArgInfo argInfo;
          argInfo.kind = argKind(kind);
          argInfo.offset = lookupNumber(arg, ".offset", 0);
          argInfo.size = lookupNumber(arg, ".size", 0);
          info.unsupported = info.unsupported || argInfo.kind == ArgKind::Unsupported;
          info.args.push_back(argInfo);
          amd_comgr_destroy_metadata(arg);
        }
        amd_comgr_destroy_metadata(args);
      }
      amd_comgr_destroy_metadata(kernel);
    }
    amd_comgr_destroy_metadata(kernels);
  }
  amd_comgr_destroy_metadata(root);
  amd_comgr_release_data(data);
}

hsa_status_t
    visitCodeObject(hsa_executable_t /*executable*/, hsa_loaded_code_object_t object, void* data) {
  auto& runtime = *static_cast<Runtime*>(data);
  hsa_agent_t agent{};
  runtime.loader.hsa_ven_amd_loader_loaded_code_object_get_info(
      object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT, &agent);
  if (agent.handle != runtime.gpu.handle) {
    return HSA_STATUS_SUCCESS;
  }
  std::uint32_t storage = 0;
  runtime.loader.hsa_ven_amd_loader_loaded_code_object_get_info(
      object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE, &storage);
  if (storage != HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY) {
    return HSA_STATUS_SUCCESS;
  }
  std::uint64_t base = 0;
  std::uint64_t size = 0;
  runtime.loader.hsa_ven_amd_loader_loaded_code_object_get_info(
      object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE, &base);
  runtime.loader.hsa_ven_amd_loader_loaded_code_object_get_info(
      object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE, &size);
  if (base != 0 && runtime.parsedCodeObjects.insert(base).second) {
    readMetadata(runtime, reinterpret_cast<const void*>(base), size);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t visitSymbol(hsa_executable_t /*executable*/,
                         hsa_agent_t /*agent*/,
                         hsa_executable_symbol_t symbol,
                         void* data) {
  auto& runtime = *static_cast<Runtime*>(data);
  hsa_symbol_kind_t kind{};
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind);
  if (kind != HSA_SYMBOL_KIND_KERNEL) {
    return HSA_STATUS_SUCCESS;
  }
  std::uint32_t length = 0;
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &length);
  std::string name(length, '\0');
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data());
  auto& info = runtime.symbols[name];
  std::uint64_t object = 0;
  hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &object);
  if (info.object != 0) {
    // (every scan visits all executables again: only a different kernel object is another one)
    if (info.object != object) {
      info.ambiguous = true;
    }
    return HSA_STATUS_SUCCESS;
  }
  info.object = object;
  if (!info.metadata) {
    hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &info.kernargSize);
    hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &info.groupSize);
    hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &info.privateSize);
  }
  return HSA_STATUS_SUCCESS;
}

} // namespace

hsa_status_t visitExecutable(hsa_executable_t executable, void* data) {
  auto& runtime = *static_cast<Runtime*>(data);
  runtime.loader.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
      executable, visitCodeObject, data);
  hsa_executable_iterate_agent_symbols(executable, runtime.gpu, visitSymbol, data);
  return HSA_STATUS_SUCCESS;
}

} // namespace device::hsaqueues
