// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_QUATERNION_QUATERNION_VANILLA_H_
#define PYVINA_QUATERNION_QUATERNION_VANILLA_H_

#include <array>

namespace pyvina
{

    namespace quaternion_vanilla
    {
        /////////////////////////////////////////////////////////
        //////  Don't forget to add '&' after const
        /////////////////////////////////////////////////////////

        std::array<double, 3> XyzArray(double x,
                                       double y,
                                       double z);
        
        double GetNorm(const std::array<double,3> &xyz);

        std::array<double, 3> Rotate(
            const std::array<double, 4> &a_quaternion,
            const std::array<double, 3> &position);

        std::array<double, 4> Conjugate(
            const std::array<double, 4> &a_quaternion);

        std::array<double, 4> FromRotationVector(
            const std::array<double, 3> &rotation_vector);

        std::array<double, 3> AsRotationVector(
            const std::array<double, 4> &a_quaternion);

        std::array<double, 4> FromAngleAxis(
            double radian,
            const std::array<double, 3> &axis);

        std::array<double, 4> Mul(
            const std::array<double, 4> &a_quaternion,
            const std::array<double, 4> &b_quaternion);

        std::array<double, 3> RotationVectorMul(
            const std::array<double, 3> &a_rotation_vector,
            const std::array<double, 3> &b_rotation_vector);

        /////////////////////////////////////////////////////////
        //////  20221018
        /////////////////////////////////////////////////////////
        template <typename T>
        double CalculateNormOfArrayn(const T &a_arrayn);

        template <typename T>
        T NormalizeArrayn(const T &a_arrayn);

    } // namespace quaternion_vanilla

} // namespace pyvina

#endif // PYVINA_QUATERNION_QUATERNION_VANILLA_H_