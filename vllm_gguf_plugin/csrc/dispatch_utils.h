#pragma once

#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, BODY)  \
  THO_DISPATCH_V2(TYPE, NAME, AT_WRAP(BODY),            \
                  torch::headeronly::ScalarType::Float, \
                  torch::headeronly::ScalarType::Half,  \
                  torch::headeronly::ScalarType::BFloat16)
