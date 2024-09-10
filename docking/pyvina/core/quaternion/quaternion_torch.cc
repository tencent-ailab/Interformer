
#include "quaternion_torch.h"

namespace pyvina
{

    torch::Tensor XyzTensor(const torch::Tensor &x,
                            const torch::Tensor &y,
                            const torch::Tensor &z)
    {
        auto xyz = torch::zeros(3);

        xyz[0] = x;
        xyz[1] = y;
        xyz[2] = z;

        return xyz;
    }

    torch::Tensor Array3ToTensor(const std::array<double, 3> &array_xyz)
    {
        auto xyz = torch::zeros(3);

        xyz[0] = array_xyz[0];
        xyz[1] = array_xyz[1];
        xyz[2] = array_xyz[2];

        return xyz;
    }

    torch::Tensor Rotate(const torch::Tensor &a_quaternion,
                         const torch::Tensor &position)
    {
        auto final_position = torch::zeros(3);

        auto b_quaternion = torch::zeros(4);
        b_quaternion[1] = position[0];
        b_quaternion[2] = position[1];
        b_quaternion[3] = position[2];

        auto c_quaternion = QuaternionMul(
            QuaternionMul(
                a_quaternion,
                b_quaternion),
            Conjugate(a_quaternion));

        final_position[0] = c_quaternion[1];
        final_position[1] = c_quaternion[2];
        final_position[2] = c_quaternion[3];

        return final_position;
    }

    torch::Tensor Conjugate(const torch::Tensor &a_quaternion)
    {
        auto conjugate_quaternion = torch::zeros(4);

        conjugate_quaternion[0] = a_quaternion[0];
        for (int i = 1; i < 4; i++)
        {
            conjugate_quaternion[i] = -a_quaternion[i];
        }

        return conjugate_quaternion;
    }

    torch::Tensor FromRotationVector(const torch::Tensor &rotation_vector)
    {
        return FromAngleAxis(
            torch::norm(rotation_vector),
            rotation_vector);
    }

    torch::Tensor AsRotationVector(const torch::Tensor &a_quaternion)
    {
        auto &&q = a_quaternion;
        auto rotation_vector = torch::zeros(3);

        auto xyz_norm = torch::sqrt(
            q[1] * q[1] +
            q[2] * q[2] +
            q[3] * q[3]);
        if (!xyz_norm.is_nonzero())
        {
            return rotation_vector;
        }

        auto theta_half = torch::atan2(xyz_norm, q[0]);
        for (int i = 0; i < 3; i++)
        {
            rotation_vector[i] = (theta_half * 2.0 * q[i + 1] / xyz_norm);
        }
        return rotation_vector;
    }

    torch::Tensor FromAngleAxis(const torch::Tensor &radian,
                                const torch::Tensor &axis)
    {
        //////////////////////////////////////////
        ////    FIXME:  How to handle 0 axis
        //////////////////////////////////////////

        auto axis_norm = torch::norm(axis);

        if (!axis_norm.is_nonzero())
        {
            auto unit_quaternion = torch::zeros(4);
            unit_quaternion[0] = 1.0;

            return unit_quaternion;
        }

        auto final_quaternion = torch::zeros(4);

        auto sin_half_radian = torch::sin(0.5 * radian);
        auto cos_half_radian = torch::cos(0.5 * radian);

        final_quaternion[0] = (cos_half_radian);
        for (int i = 0; i < 3; i++)
        {
            final_quaternion[i + 1] = (sin_half_radian * axis[i] / axis_norm);
        }

        return final_quaternion;
    }

    torch::Tensor QuaternionMul(const torch::Tensor &a_quaternion,
                                const torch::Tensor &b_quaternion)
    {
        auto final_quaternion = torch::zeros(4);

        auto w1 = a_quaternion[0];
        auto i1 = a_quaternion[1];
        auto j1 = a_quaternion[2];
        auto k1 = a_quaternion[3];

        auto w2 = b_quaternion[0];
        auto i2 = b_quaternion[1];
        auto j2 = b_quaternion[2];
        auto k2 = b_quaternion[3];

        final_quaternion[0] = (w1 * w2 - i1 * i2 - j1 * j2 - k1 * k2);
        final_quaternion[1] = (w1 * i2 + i1 * w2 + j1 * k2 - k1 * j2);
        final_quaternion[2] = (w1 * j2 - i1 * k2 + j1 * w2 + k1 * i2);
        final_quaternion[3] = (w1 * k2 + i1 * j2 - j1 * i2 + k1 * w2);

        return final_quaternion;
    }

    torch::Tensor RotationVectorMul(
        const torch::Tensor &a_rotation_vector,
        const torch::Tensor &b_rotation_vector)
    {
        auto a_quaternion = FromRotationVector(a_rotation_vector);
        auto b_quaternion = FromRotationVector(b_rotation_vector);

        auto c_quaternion = QuaternionMul(
            a_quaternion,
            b_quaternion);

        auto c_rotation_vector = AsRotationVector(
            c_quaternion);

        return c_rotation_vector;
    }

} // namespace pyvina