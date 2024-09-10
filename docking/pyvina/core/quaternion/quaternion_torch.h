#ifndef PYVINA_QUATERNION_QUATERNION_TORCH_H_
#define PYVINA_QUATERNION_QUATERNION_TORCH_H_

#include <torch/torch.h>

#include "../array.hpp"

namespace pyvina
{

    torch::Tensor XyzTensor(const torch::Tensor &x,
                            const torch::Tensor &y,
                            const torch::Tensor &z);
    torch::Tensor Array3ToTensor(const std::array<double, 3> &array_xyz);

    torch::Tensor Rotate(const torch::Tensor &a_quaternion,
                         const torch::Tensor &position);

    torch::Tensor Conjugate(const torch::Tensor &a_quaternion);

    torch::Tensor FromRotationVector(const torch::Tensor &rotation_vector);
    torch::Tensor AsRotationVector(const torch::Tensor &a_quaternion);

    torch::Tensor FromAngleAxis(const torch::Tensor &radian,
                                const torch::Tensor &axis);

    torch::Tensor QuaternionMul(const torch::Tensor &a_quaternion,
                                const torch::Tensor &b_quaternion);

    torch::Tensor RotationVectorMul(const torch::Tensor &a_rotation_vector,
                                    const torch::Tensor &b_rotation_vector);

} // namespace pyvina

#endif // PYVINA_QUATERNION_QUATERNION_TORCH_H_