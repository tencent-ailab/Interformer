
#include "tensor_tests.h"

torch::Tensor AddOne(const torch::Tensor &x)
{
    return x + 1;
}