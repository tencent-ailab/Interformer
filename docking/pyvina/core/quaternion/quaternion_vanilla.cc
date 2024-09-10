// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#include "quaternion_vanilla.h"

#include <cmath>
// #include <math.h>
#include <stdexcept>

#include"pyvina/core/wrappers_for_exceptions/common.h"

namespace pyvina
{
    namespace quaternion_vanilla
    {

        std::array<double, 3> XyzArray(double x,
                                       double y,
                                       double z)
        {
            std::array<double, 3> xyz = {x, y, z};
            return xyz;
        }

        double GetNorm(const std::array<double, 3> &xyz)
        {
            double norm2 = (xyz[0] * xyz[0] +
                            xyz[1] * xyz[1] +
                            xyz[2] * xyz[2]);
            return std::sqrt(norm2);
        }

        std::array<double, 3> Rotate(
            const std::array<double, 4> &a_quaternion,
            const std::array<double, 3> &position)
        {
            std::array<double, 3> final_position = {
                0.0, 0.0, 0.0};

            std::array<double, 4> b_quaternion = {
                0.0,
                position[0],
                position[1],
                position[2]};

            auto c_quaternion = Mul(
                Mul(
                    a_quaternion,
                    b_quaternion),
                Conjugate(a_quaternion));

            final_position[0] = c_quaternion[1];
            final_position[1] = c_quaternion[2];
            final_position[2] = c_quaternion[3];

            return final_position;
        }

        std::array<double, 4> Conjugate(
            const std::array<double, 4> &a_quaternion)
        {
            auto b_quaternion = a_quaternion;
            for (int i = 1; i < 4; i++)
            {
                b_quaternion[i] *= -1.0;
            }
            return b_quaternion;
        }

        std::array<double, 4> FromRotationVector(
            const std::array<double, 3> &rotation_vector)
        {
            return FromAngleAxis(
                GetNorm(rotation_vector),
                rotation_vector);
        }

        std::array<double, 3> AsRotationVector(
            const std::array<double, 4> &a_quaternion)
        {
            auto &&q = a_quaternion;
            std::array<double, 3> rotation_vector = {
                0.0, 0.0, 0.0};

            auto xyz_norm = std::sqrt(
                q[1] * q[1] +
                q[2] * q[2] +
                q[3] * q[3]);

            if (xyz_norm == 0.0)
            {
                return rotation_vector;
            }

            auto theta_half = atan2(xyz_norm, q[0]);
            for (int i = 0; i < 3; i++)
            {
                rotation_vector[i] = (theta_half * 2.0 * q[i + 1] / xyz_norm);
            }
            return rotation_vector;
        }

        std::array<double, 4> FromAngleAxis(
            double radian,
            const std::array<double, 3> &axis)
        {
            double axis_norm = GetNorm(axis);

            if (axis_norm == 0.0)
            {
                std::array<double, 4> unit_quaternion = {
                    1.0, 0.0, 0.0, 0.0};
                return unit_quaternion;
            }

            std::array<double, 4> final_quaternion = {
                0.0, 0.0, 0.0, 0.0};
            auto sin_half_radian = sin(0.5 * radian);
            auto cos_half_radian = cos(0.5 * radian);

            final_quaternion[0] = (cos_half_radian);
            for (int i = 0; i < 3; i++)
            {
                final_quaternion[i + 1] = (sin_half_radian * axis[i] / axis_norm);
            }

            if (pyvina::core::wrappers_for_exceptions::IsNanInArray1D(final_quaternion)) {
              pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(

                  " :: FromAngleAxis :: quaternion :: "

                  + pyvina::core::wrappers_for_exceptions::GetStrFromArray1D(final_quaternion) + " :: \n" +

                  " :: radian :: "

                  + std::to_string(radian) + " :: \n" +

                  " :: axis :: "

                  + pyvina::core::wrappers_for_exceptions::GetStrFromArray1D(axis) + " :: \n"

              );
            }

            return final_quaternion;
        }

        std::array<double, 4> Mul(
            const std::array<double, 4> &a_quaternion,
            const std::array<double, 4> &b_quaternion)
        {
            std::array<double, 4> final_quaternion = {
                0.0, 0.0, 0.0, 0.0};

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

        std::array<double, 3> RotationVectorMul(
            const std::array<double, 3> &a_rotation_vector,
            const std::array<double, 3> &b_rotation_vector)
        {
            auto a_quaternion = FromRotationVector(a_rotation_vector);
            auto b_quaternion = FromRotationVector(b_rotation_vector);

            auto c_quaternion = Mul(
                a_quaternion,
                b_quaternion);

            auto c_rotation_vector = AsRotationVector(
                c_quaternion);

            return c_rotation_vector;
        }

        /////////////////////////////////////////////////////////
        //////  20221018
        /////////////////////////////////////////////////////////
        template <typename T>
        double CalculateNormOfArrayn(const T &a_arrayn) {
          double norm_square = 0.0;

          for (int i = 0; i < a_arrayn.size(); i++) {
            norm_square += (a_arrayn[i] * a_arrayn[i]);
          }

          return std::sqrt(norm_square);
        }

        template <typename T>
        T NormalizeArrayn(const T &a_arrayn) {
          double norm_arrayn = CalculateNormOfArrayn(a_arrayn);

          if (norm_arrayn == 0.0) {
            pyvina::core::wrappers_for_exceptions::Throw1ExceptionAfterPrinting(
                " :: .NormalizeArrayn() but norm == 0.0 :: \n");
          }

          T normalized_arrayn = T(a_arrayn);
          for (int i = 0; i < normalized_arrayn.size(); i++) {
            normalized_arrayn[i] /= norm_arrayn;
          }

          return normalized_arrayn;
        }

        template double CalculateNormOfArrayn(const std::array<double, 3> &);
        template double CalculateNormOfArrayn(const std::array<double, 4> &);

        template std::array<double, 3> NormalizeArrayn(const std::array<double, 3> &);
        template std::array<double, 4> NormalizeArrayn(const std::array<double, 4> &);

    } // namespace quaternion_vanilla
} // namespace pyvina
