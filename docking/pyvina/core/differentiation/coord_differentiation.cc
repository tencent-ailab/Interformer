// Copyright 2021 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

// #include <math.h>
#include <cmath>

#include "coord_differentiation.h"

namespace pyvina
{

    namespace differentiation
    {

        /////////////////////////////////////////////////////////
        /////
        /////  For calculating the derivatives of final_coord in this formula:
        /////  $$
        /////  final_coord = origin_coord + rotate( quaternion, delta_vector )
        /////  $$
        /////  with respect to '6+K'
        /////
        /////  check details in .h file
        /////
        /////////////////////////////////////////////////////////
        std::vector<std::vector<double>> CoordDiffCore_AddRotatedVectorDerivatives(

            int num_degrees_of_freedom,

            double d_x,
            double d_y,
            double d_z,

            double q_w,
            double q_i,
            double q_j,
            double q_k,

            const std::vector<std::vector<double>> &origin_derivatives,
            const std::vector<std::vector<double>> &quaternion_derivatives

        )
        {
            auto derivatives_return = std::vector<std::vector<double>>(
                3,
                std::vector<double>(num_degrees_of_freedom, 0.0));

            auto derivatives_v_xyz_to_q_wijk = std::vector<std::vector<double>>(
                3,
                std::vector<double>(4, 0.0));

            //////////////////////////////////////////////
            /////  v_x to frame_quaternion ( q_w, q_i, q_j, q_k )
            //////////////////////////////////////////////
            derivatives_v_xyz_to_q_wijk[0][0] = 2 * d_x * q_w - 2 * d_y * q_k + 2 * d_z * q_j;
            derivatives_v_xyz_to_q_wijk[0][1] = 2 * d_x * q_i + 2 * d_y * q_j + 2 * d_z * q_k;
            derivatives_v_xyz_to_q_wijk[0][2] = -2 * d_x * q_j + 2 * d_y * q_i + 2 * d_z * q_w;
            derivatives_v_xyz_to_q_wijk[0][3] = -2 * d_x * q_k - 2 * d_y * q_w + 2 * d_z * q_i;

            //////////////////////////////////////////////
            /////  v_y to frame_quaternion ( q_w, q_i, q_j, q_k )
            //////////////////////////////////////////////
            derivatives_v_xyz_to_q_wijk[1][0] = 2 * d_x * q_k + 2 * d_y * q_w - 2 * d_z * q_i;
            derivatives_v_xyz_to_q_wijk[1][1] = 2 * d_x * q_j - 2 * d_y * q_i - 2 * d_z * q_w;
            derivatives_v_xyz_to_q_wijk[1][2] = 2 * d_x * q_i + 2 * d_y * q_j + 2 * d_z * q_k;
            derivatives_v_xyz_to_q_wijk[1][3] = 2 * d_x * q_w - 2 * d_y * q_k + 2 * d_z * q_j;

            //////////////////////////////////////////////
            /////  v_z to frame_quaternion ( q_w, q_i, q_j, q_k )
            //////////////////////////////////////////////
            derivatives_v_xyz_to_q_wijk[2][0] = -2 * d_x * q_j + 2 * d_y * q_i + 2 * d_z * q_w;
            derivatives_v_xyz_to_q_wijk[2][1] = 2 * d_x * q_k + 2 * d_y * q_w - 2 * d_z * q_i;
            derivatives_v_xyz_to_q_wijk[2][2] = -2 * d_x * q_w + 2 * d_y * q_k - 2 * d_z * q_j;
            derivatives_v_xyz_to_q_wijk[2][3] = 2 * d_x * q_i + 2 * d_y * q_j + 2 * d_z * q_k;

            ///////////////////////////////////////////////////////////
            /////  d({heavy Atom}_x)/d( {'6+K'} ) = d(o_x)/d({'6+K'}) + d(v_x)/d({'6+K'})
            /////  d({heavy Atom}_y)/d( {'6+K'} ) = d(o_y)/d({'6+K'}) + d(v_y)/d({'6+K'})
            /////  d({heavy Atom}_z)/d( {'6+K'} ) = d(o_z)/d({'6+K'}) + d(v_z)/d({'6+K'})
            ///////////////////////////////////////////////////////////
            for (int degree_index = 0;
                 degree_index < num_degrees_of_freedom;
                 degree_index++)
            {
                /////  += d(o_xyz)/d({'6+K'})
                //////////////////////////////////////////////////////////////////////
                for (int xyz_index = 0; xyz_index < 3; xyz_index++)
                {
                    derivatives_return[xyz_index][degree_index] += (origin_derivatives[xyz_index][degree_index]);
                }

                /////  += d(v_xyz)/d({'6+K'})
                //////////////////////////////////////////////////////////////////////
                /////
                /////  d(v_xyz)/d({'6+K'})
                /////  =
                /////  d(v_xyz)/d({'6+K'}) +
                /////  d(v_xyz)/d(q_w) * d(q_w)/d({'6+K'}) +
                /////  d(v_xyz)/d(q_i) * d(q_i)/d({'6+K'}) +
                /////  d(v_xyz)/d(q_j) * d(q_j)/d({'6+K'}) +
                /////  d(v_xyz)/d(q_k) * d(q_k)/d({'6+K'})
                /////  =
                /////  d(v_xyz)/d(q_w) * d(q_w)/d({'6+K'}) +
                /////  d(v_xyz)/d(q_i) * d(q_i)/d({'6+K'}) +
                /////  d(v_xyz)/d(q_j) * d(q_j)/d({'6+K'}) +
                /////  d(v_xyz)/d(q_k) * d(q_k)/d({'6+K'})
                /////
                //////////////////////////////////////////////////////////////////////
                /////  q_index ( 0, 1, 2, 3 ) -> quaternion ( q_w, q_i, q_j, q_k )
                //////////////////////////////////////////////////////////////////////
                for (
                    int q_index = 0;
                    q_index < 4;
                    q_index++

                )
                {
                    for (int xyz_index = 0; xyz_index < 3; xyz_index++)
                    {
                        derivatives_return[xyz_index][degree_index] += (derivatives_v_xyz_to_q_wijk[xyz_index][q_index] *
                                                                        quaternion_derivatives[q_index][degree_index]);
                    }
                }
            }

            return derivatives_return;
        }

        /////////////////////////////////////////////////////////
        /////
        /////  heavy_atom_coords ( ha_x , ha_y , ha_z ) derivatives to '6+K'
        /////  check details in .h file
        /////
        /////////////////////////////////////////////////////////
        std::vector<std::vector<double>> CoordDiffCore_GetHeavyAtomCoordDerivatives(

            int num_degrees_of_freedom,

            ///// frame_position
            double d_x,
            double d_y,
            double d_z,

            ///// frame_derivatives
            double q_w,
            double q_i,
            double q_j,
            double q_k,

            const std::vector<std::vector<double>> &frame_position_derivatives,
            const std::vector<std::vector<double>> &frame_quaternion_derivatives

        )
        {
            return CoordDiffCore_AddRotatedVectorDerivatives(

                num_degrees_of_freedom,

                d_x,
                d_y,
                d_z,

                q_w,
                q_i,
                q_j,
                q_k,

                frame_position_derivatives,
                frame_quaternion_derivatives

            );

            // auto derivatives_return = std::vector<std::vector<double>>(
            //     3,
            //     std::vector<double>(num_degrees_of_freedom, 0.0));

            // auto derivatives_v_xyz_to_q_wijk = std::vector<std::vector<double>>(
            //     3,
            //     std::vector<double>(4, 0.0));

            // //////////////////////////////////////////////
            // /////  v_x to frame_quaternion ( q_w, q_i, q_j, q_k )
            // //////////////////////////////////////////////
            // derivatives_v_xyz_to_q_wijk[0][0] = 2 * d_x * q_w - 2 * d_y * q_k + 2 * d_z * q_j;
            // derivatives_v_xyz_to_q_wijk[0][1] = 2 * d_x * q_i + 2 * d_y * q_j + 2 * d_z * q_k;
            // derivatives_v_xyz_to_q_wijk[0][2] = -2 * d_x * q_j + 2 * d_y * q_i + 2 * d_z * q_w;
            // derivatives_v_xyz_to_q_wijk[0][3] = -2 * d_x * q_k - 2 * d_y * q_w + 2 * d_z * q_i;

            // //////////////////////////////////////////////
            // /////  v_y to frame_quaternion ( q_w, q_i, q_j, q_k )
            // //////////////////////////////////////////////
            // derivatives_v_xyz_to_q_wijk[1][0] = 2 * d_x * q_k + 2 * d_y * q_w - 2 * d_z * q_i;
            // derivatives_v_xyz_to_q_wijk[1][1] = 2 * d_x * q_j - 2 * d_y * q_i - 2 * d_z * q_w;
            // derivatives_v_xyz_to_q_wijk[1][2] = 2 * d_x * q_i + 2 * d_y * q_j + 2 * d_z * q_k;
            // derivatives_v_xyz_to_q_wijk[1][3] = 2 * d_x * q_w - 2 * d_y * q_k + 2 * d_z * q_j;

            // //////////////////////////////////////////////
            // /////  v_z to frame_quaternion ( q_w, q_i, q_j, q_k )
            // //////////////////////////////////////////////
            // derivatives_v_xyz_to_q_wijk[2][0] = -2 * d_x * q_j + 2 * d_y * q_i + 2 * d_z * q_w;
            // derivatives_v_xyz_to_q_wijk[2][1] = 2 * d_x * q_k + 2 * d_y * q_w - 2 * d_z * q_i;
            // derivatives_v_xyz_to_q_wijk[2][2] = -2 * d_x * q_w + 2 * d_y * q_k - 2 * d_z * q_j;
            // derivatives_v_xyz_to_q_wijk[2][3] = 2 * d_x * q_i + 2 * d_y * q_j + 2 * d_z * q_k;

            // ///////////////////////////////////////////////////////////
            // /////  d({heavy Atom}_x)/d( {'6+K'} ) = d(o_x)/d({'6+K'}) + d(v_x)/d({'6+K'})
            // /////  d({heavy Atom}_y)/d( {'6+K'} ) = d(o_y)/d({'6+K'}) + d(v_y)/d({'6+K'})
            // /////  d({heavy Atom}_z)/d( {'6+K'} ) = d(o_z)/d({'6+K'}) + d(v_z)/d({'6+K'})
            // ///////////////////////////////////////////////////////////
            // for (int degree_index = 0;
            //      degree_index < num_degrees_of_freedom;
            //      degree_index++)
            // {
            //     /////  += d(o_xyz)/d({'6+K'})
            //     //////////////////////////////////////////////////////////////////////
            //     for (int xyz_index = 0; xyz_index < 3; xyz_index++)
            //     {
            //         derivatives_return[xyz_index][degree_index] += (frame_position_derivatives[xyz_index][degree_index]);
            //     }

            //     /////  += d(v_xyz)/d({'6+K'})
            //     //////////////////////////////////////////////////////////////////////
            //     /////
            //     /////  d(v_xyz)/d({'6+K'})
            //     /////  =
            //     /////  d(v_xyz)/d({'6+K'}) +
            //     /////  d(v_xyz)/d(q_w) * d(q_w)/d({'6+K'}) +
            //     /////  d(v_xyz)/d(q_i) * d(q_i)/d({'6+K'}) +
            //     /////  d(v_xyz)/d(q_j) * d(q_j)/d({'6+K'}) +
            //     /////  d(v_xyz)/d(q_k) * d(q_k)/d({'6+K'})
            //     /////  =
            //     /////  d(v_xyz)/d(q_w) * d(q_w)/d({'6+K'}) +
            //     /////  d(v_xyz)/d(q_i) * d(q_i)/d({'6+K'}) +
            //     /////  d(v_xyz)/d(q_j) * d(q_j)/d({'6+K'}) +
            //     /////  d(v_xyz)/d(q_k) * d(q_k)/d({'6+K'})
            //     /////
            //     //////////////////////////////////////////////////////////////////////
            //     /////  q_index ( 0, 1, 2, 3 ) -> quaternion ( q_w, q_i, q_j, q_k )
            //     //////////////////////////////////////////////////////////////////////
            //     for (
            //         int q_index = 0;
            //         q_index < 4;
            //         q_index++

            //     )
            //     {
            //         for (int xyz_index = 0; xyz_index < 3; xyz_index++)
            //         {
            //             derivatives_return[xyz_index][degree_index] += (derivatives_v_xyz_to_q_wijk[xyz_index][q_index] *
            //                                                             frame_quaternion_derivatives[q_index][degree_index]);
            //         }
            //     }
            // }

            // return derivatives_return;
        }

        /////////////////////////////////////////////////////////
        /////
        /////  root_frame_origin_position ( o_x, o_y, o_z ) derivatives to '6+K'
        /////  check details in .h file
        /////
        /////////////////////////////////////////////////////////
        std::vector<std::vector<double>> CoordDiffCore_GetRootFramePositionDerivatives(

            int num_degrees_of_freedom
            // int num_degrees_of_freedom,

            // double x,
            // double y,
            // double z,

            // double i,
            // double j,
            // double k

        )
        {
            auto derivatives_return = std::vector<std::vector<double>>(
                3,
                std::vector<double>(num_degrees_of_freedom, 0.0));

            // index of (x,y,z) of '6+K' -> (0,1,2)

            // [0, 0] -> [origin_position_x, x of '6+K']
            derivatives_return[0][0] = 1.0;
            // [0, 0] -> [origin_position_y, y of '6+K']
            derivatives_return[1][1] = 1.0;
            // [0, 0] -> [origin_position_z, z of '6+K']
            derivatives_return[2][2] = 1.0;

            return derivatives_return;
        }

        /////////////////////////////////////////////////////////
        /////
        /////  root_frame_quaternion ( q_w, q_i, q_j, q_k ) derivatives to '6+K'
        /////  check details in .h file
        /////
        /////////////////////////////////////////////////////////
        std::vector<std::vector<double>> CoordDiffCore_GetRootFrameQuaternionDerivatives(

            int num_degrees_of_freedom,

            ///// '3' from '6+K'
            double x,
            double y,
            double z,

            ///// '3' from '6+K'
            double i,
            double j,
            double k

        )
        {
            auto derivatives_return = std::vector<std::vector<double>>(
                4,
                std::vector<double>(num_degrees_of_freedom, 0.0));

            auto r = sqrt(i * i + j * j + k * k);
            auto sin_half_r = sin(r / 2.0);
            auto cos_half_r = cos(r / 2.0);

            // index of (i,j,k) of '6+K' -> (3,4,5)

            // quaternion_w -> (i,j,k) of '6+K'
            derivatives_return[0][3] = -1.0 / 2.0 * i * sin_half_r / r;
            derivatives_return[0][4] = -1.0 / 2.0 * j * sin_half_r / r;
            derivatives_return[0][5] = -1.0 / 2.0 * k * sin_half_r / r;

            // quaternion_i -> (i,j,k) of '6+K'
            derivatives_return[1][3] = ((1.0 / 2.0) * std::pow(i, 2) * (cos_half_r * r - 2 * sin_half_r) + std::pow(r, 2) * sin_half_r) / std::pow(r, 3);
            derivatives_return[1][4] = (1.0 / 2.0) * i * j * (cos_half_r * r - 2 * sin_half_r) / std::pow(r, 3);
            derivatives_return[1][5] = (1.0 / 2.0) * i * k * (cos_half_r * r - 2 * sin_half_r) / std::pow(r, 3);

            // quaternion_j -> (i,j,k) of '6+K'
            derivatives_return[2][3] = (1.0 / 2.0) * i * j * (cos_half_r * r - 2 * sin_half_r) / std::pow(r, 3);
            derivatives_return[2][4] = ((1.0 / 2.0) * std::pow(j, 2) * (cos_half_r * r - 2 * sin_half_r) + std::pow(r, 2) * sin_half_r) / std::pow(r, 3);
            derivatives_return[2][5] = (1.0 / 2.0) * j * k * (cos_half_r * r - 2 * sin_half_r) / std::pow(r, 3);

            // quaternion_k -> (i,j,k) of '6+K'
            derivatives_return[3][3] = (1.0 / 2.0) * i * k * (cos_half_r * r - 2 * sin_half_r) / std::pow(r, 3);
            derivatives_return[3][4] = (1.0 / 2.0) * j * k * (cos_half_r * r - 2 * sin_half_r) / std::pow(r, 3);
            derivatives_return[3][5] = ((1.0 / 2.0) * std::pow(k, 2) * (cos_half_r * r - 2 * sin_half_r) + std::pow(r, 2) * sin_half_r) / std::pow(r, 3);

            return derivatives_return;
        }

        /////////////////////////////////////////////////////////
        /////
        /////  non_root_frame_origin_position ( o_x, o_y, o_z ) derivatives to '6+K'
        /////  check details in .h file
        /////
        /////////////////////////////////////////////////////////
        std::vector<std::vector<double>> CoordDiffCore_GetNonRootFramePositionDerivatives(

            int num_degrees_of_freedom,

            ///// rotor X->Y
            double rotor_x,
            double rotor_y,
            double rotor_z,

            ///// current_frame_quaternion
            double q_w,
            double q_i,
            double q_j,
            double q_k,

            ///// Here, x from 'current_frame_heavy_atom_x_coord_derivatives' means X from rotor X->Y
            ///// rather than the x coord from ( x, y, z )
            const std::vector<std::vector<double>> &current_frame_heavy_atom_x_coord_derivatives,
            const std::vector<std::vector<double>> &current_frame_quaternion_derivatives

        )
        {
            return CoordDiffCore_AddRotatedVectorDerivatives(

                num_degrees_of_freedom,

                rotor_x,
                rotor_y,
                rotor_z,

                q_w,
                q_i,
                q_j,
                q_k,

                current_frame_heavy_atom_x_coord_derivatives,
                current_frame_quaternion_derivatives

            );
        }

        /////////////////////////////////////////////////////////
        /////
        /////  non_root_frame_quaternion
        /////  ( next_q_w, next_q_i, next_q_j, next_q_k )
        /////  derivatives to '6+K'
        /////
        /////  check details in .h file
        /////
        /////////////////////////////////////////////////////////
        std::vector<std::vector<double>> CoordDiffCore_GetNonRootFrameQuaternionDerivatives(

            int num_degrees_of_freedom,

            ///// the index of the next frame (e.g. root_frame_index is 0, frame_1_index is 1)
            int k_next_frame,

            ///// the angle rotated around the axis
            double radian,

            ///// rotor X->Y
            double rotor_x,
            double rotor_y,
            double rotor_z,

            ///// current_frame_quaternion
            double q_w,
            double q_i,
            double q_j,
            double q_k,

            const std::vector<std::vector<double>> &current_frame_quaternion_derivatives

        )
        {
            auto derivatives_return = std::vector<std::vector<double>>(
                4,
                std::vector<double>(num_degrees_of_freedom, 0.0));

            auto derivatives_quaternion_next_to_current = std::vector<std::vector<double>>(
                4,
                std::vector<double>(4, 0.0));

            double sin_radian_half = std::sin(0.5 * radian);
            double cos_radian_half = std::cos(0.5 * radian);

            //////////////////////////////////////////////
            /////  next ( q_w ) to current ( q_w, q_i, q_j, q_k )
            /////
            /////  derivatives next_frame_quaternion_q_w
            /////  to current_frame_quaternion ( q_w, q_i, q_j, q_k )
            //////////////////////////////////////////////
            derivatives_quaternion_next_to_current[0][0] = cos_radian_half;
            derivatives_quaternion_next_to_current[0][1] = -1.0 * rotor_x * sin_radian_half;
            derivatives_quaternion_next_to_current[0][2] = -1.0 * rotor_y * sin_radian_half;
            derivatives_quaternion_next_to_current[0][3] = -1.0 * rotor_z * sin_radian_half;

            //////////////////////////////////////////////
            /////  next ( q_i ) to current ( q_w, q_i, q_j, q_k )
            /////
            /////  derivatives next_frame_quaternion_q_i
            /////  to current_frame_quaternion ( q_w, q_i, q_j, q_k )
            //////////////////////////////////////////////
            derivatives_quaternion_next_to_current[1][0] = 1.0 * rotor_x * sin_radian_half;
            derivatives_quaternion_next_to_current[1][1] = cos_radian_half;
            derivatives_quaternion_next_to_current[1][2] = 1.0 * rotor_z * sin_radian_half;
            derivatives_quaternion_next_to_current[1][3] = -1.0 * rotor_y * sin_radian_half;

            //////////////////////////////////////////////
            /////  next ( q_j ) to current ( q_w, q_i, q_j, q_k )
            /////
            /////  derivatives next_frame_quaternion_q_j
            /////  to current_frame_quaternion ( q_w, q_i, q_j, q_k )
            //////////////////////////////////////////////
            derivatives_quaternion_next_to_current[2][0] = 1.0 * rotor_y * sin_radian_half;
            derivatives_quaternion_next_to_current[2][1] = -1.0 * rotor_z * sin_radian_half;
            derivatives_quaternion_next_to_current[2][2] = cos_radian_half;
            derivatives_quaternion_next_to_current[2][3] = 1.0 * rotor_x * sin_radian_half;

            //////////////////////////////////////////////
            /////  next ( q_k ) to current ( q_w, q_i, q_j, q_k )
            /////
            /////  derivatives next_frame_quaternion_q_k
            /////  to current_frame_quaternion ( q_w, q_i, q_j, q_k )
            //////////////////////////////////////////////
            derivatives_quaternion_next_to_current[3][0] = 1.0 * rotor_z * sin_radian_half;
            derivatives_quaternion_next_to_current[3][1] = 1.0 * rotor_y * sin_radian_half;
            derivatives_quaternion_next_to_current[3][2] = -1.0 * rotor_x * sin_radian_half;
            derivatives_quaternion_next_to_current[3][3] = cos_radian_half;

            /////////////////////////////////////////////////////////
            /////
            /////  $$
            /////  next_{quaternion} = multiply( current_{quaternion} , fromAngleAxis( radian, rotor_{xyz} ) )
            /////  $$
            /////
            /////  $$
            /////  d( next )/d( '6+K' )
            /////  =
            /////  d( next )/d( '6+K' ) +
            /////  d( next )/d( q_w ) * d( q_w )/d( '6+K' ) +
            /////  d( next )/d( q_i ) * d( q_i )/d( '6+K' ) +
            /////  d( next )/d( q_j ) * d( q_j )/d( '6+K' ) +
            /////  d( next )/d( q_k ) * d( q_k )/d( '6+K' )
            /////  $$
            /////
            /////////////////////////////////////////////////////////

            /////////////////////////////////////////////////////////
            /////  $$
            /////  d( next )/d( '6+K' )
            /////  +=
            /////  d( next )/d( '6+K' )
            /////  $$
            /////
            /////  For the second 'd( next )/d( '6+K' )'
            /////  The only non-zero second 'd( next )/d( '6+K' )'
            /////  is 'd( next )/d( radian )'
            /////
            /////  $$
            /////  d( next )/d( radian )
            /////  +=
            /////  d( next )/d( radian )
            /////  $$
            /////////////////////////////////////////////////////////
            /////
            /////  5 + index_frame = the index of 'radian' from '6+K'
            /////
            /////  e.g.
            /////  root_frame index is (5 + 0) = 5 (and therefore skipped)
            /////  frame_1 is (5 + 1) = 6
            /////
            /////////////////////////////////////////////////////////
            derivatives_return[0][5 + k_next_frame] += (-0.5 * cos_radian_half * q_i * rotor_x - 0.5 * cos_radian_half * q_j * rotor_y - 0.5 * cos_radian_half * q_k * rotor_z - 0.5 * q_w * sin_radian_half);
            derivatives_return[1][5 + k_next_frame] += (0.5 * cos_radian_half * q_j * rotor_z - 0.5 * cos_radian_half * q_k * rotor_y + 0.5 * cos_radian_half * q_w * rotor_x - 0.5 * q_i * sin_radian_half);
            derivatives_return[2][5 + k_next_frame] += (-0.5 * cos_radian_half * q_i * rotor_z + 0.5 * cos_radian_half * q_k * rotor_x + 0.5 * cos_radian_half * q_w * rotor_y - 0.5 * q_j * sin_radian_half);
            derivatives_return[3][5 + k_next_frame] += (0.5 * cos_radian_half * q_i * rotor_y - 0.5 * cos_radian_half * q_j * rotor_x + 0.5 * cos_radian_half * q_w * rotor_z - 0.5 * q_k * sin_radian_half);

            /////////////////////////////////////////////////////////
            /////  $$
            /////  d( next )/d( '6+K' )
            /////  +=
            /////  d( next )/d( q_w ) * d( q_w )/d( '6+K' ) +
            /////  d( next )/d( q_i ) * d( q_i )/d( '6+K' ) +
            /////  d( next )/d( q_j ) * d( q_j )/d( '6+K' ) +
            /////  d( next )/d( q_k ) * d( q_k )/d( '6+K' )
            /////  $$
            /////////////////////////////////////////////////////////
            for (int index_next = 0;
                 index_next < 4;
                 index_next++)
            {
                for (int index_degree = 0;
                     index_degree < num_degrees_of_freedom;
                     index_degree++)
                {
                    for (int index_current = 0;
                         index_current < 4;
                         index_current++)
                    {
                        derivatives_return[index_next][index_degree] += (derivatives_quaternion_next_to_current[index_next][index_current] *
                                                                         current_frame_quaternion_derivatives[index_current][index_degree]);
                    }
                }
            }

            return derivatives_return;
        }

    } // namespace differentiation

} // namespace pyvina
