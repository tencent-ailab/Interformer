// Copyright 2021 Tencent Inc. All rights reserved.
//
// Author: ruiyuanqian@tencent.com (ruiyuanqian)
//
// 文件注释
// ...

#ifndef PYVINA_DIFFERENTIATION_COORD_DIFFERENTIATION_H_
#define PYVINA_DIFFERENTIATION_COORD_DIFFERENTIATION_H_

#include <vector>

namespace pyvina
{

    namespace differentiation
    {

        /////////////////////////////////////////////////////////
        /////  FORMULAS FOR BOTH ROOT FRAMES AND NON-ROOT FRAMES
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        /////
        /////  For calculating the derivatives of final_coord in this formula:
        /////
        /////  $$
        /////  final_coord = origin_coord + rotate( quaternion, delta_vector )
        /////  $$
        /////
        /////  Here, rotate( quaternion , delta_vector ) is a func to
        /////  rotate delta_vector ( d_x , d_y , d_z ) based on
        /////  quaternion ( q_w , q_i , q_j , q_k )
        /////
        /////  This func returns the derivatives of final_coord ( f_x , f_y , f_z ) to '6+K'
        /////  based on:
        /////
        /////  1. origin_coord ( o_x, o_y, o_z ) derivatives to '6+K'
        /////  2. quaternion ( q_w, q_i, q_j, q_k ) derivatives to '6+K'
        /////  3. quaternion ( q_w, q_i, q_j, q_k )
        /////  4. delta vector ( d_x, d_y, d_z )
        /////  5. num_'6+K' ( that is, num_degrees_of_freedom )
        /////
        /////////////////////////////////////////////////////////
        /////
        /////  Formula
        /////  $$
        /////  final_coord = origin_coord + rotate( quaternion, delta_vector )
        /////  $$
        /////  can be applied to many conditions, such as:
        /////
        /////  1. intra-frame
        /////     final_coord   ->  one_heavy_atom coord
        /////     origin_coord  ->  frame_orgin coord
        /////      ( CoordDiffCore_GetHeavyAtomCoordDerivatives() )
        /////
        /////  2. inter-frame
        /////     final_coord   ->  next_frame_position_Y coord
        /////     origin_coord  ->  current_frame_X
        /////      ( X->Y is the rotor between current and next frames )
        /////      ( CoordDiffCore_GetNonRootFramePositionDerivatives() )
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

        );

        /////////////////////////////////////////////////////////
        /////
        /////  Return a matrix with shape ( 3 , '6+K' )
        /////
        /////  This matrix means:
        /////  the derivatives of root_frame absolute_heavy_atom_coords ( ha_x , ha_y , ha_z )
        /////
        /////  To calculate these heavy_atom_coords derivatives,
        /////  these 2 intermediate variables are used:
        /////
        /////  1. frame_position
        /////  2. frame_quaternion
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

        );

        /////////////////////////////////////////////////////////
        /////  ROOT FRAME FORMULAS
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        /////
        /////  Return a matrix with shape ( 3 , '6+K' )
        /////
        /////  Here '3' is for root_frame_origin_position ( origin_x, origin_y, origin_z )
        /////  And '6+K' is the num_degrees_of_freedom
        /////
        /////////////////////////////////////////////////////////
        /////
        /////  This matrix means:
        /////  the derivatives of root_frame_origin_position ( o_x, o_y, o_z )
        /////  with respect to '6+K'
        /////
        /////  '6+K' -> conformation:
        /////
        /////  3  ->  position (x,y,z)
        /////  +
        /////  3  ->  rotation vector (i,j,k)
        /////  +
        /////  K  ->  num of torsions ( num_of_torsions = num_of_frames - 1 )
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

        );

        /////////////////////////////////////////////////////////
        /////
        /////  Return a matrix with shape ( 4 , '6+K' )
        /////
        /////  This matrix means the derivatives of root_frame_quaternion ( q_w, q_i, q_j, q_k )
        /////  with respect to '6+K'
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

        );

        /////////////////////////////////////////////////////////
        /////  NON-ROOT FRAME FORMULAS
        /////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////
        /////
        /////  non-root frame version of func CoordDiffCore_GetRootFramePositionDerivatives()
        /////
        /////////////////////////////////////////////////////////
        /////
        /////  X in current_frame -> Y in next_frame
        /////  vector X->Y is the rotor for next_frame (the rotor between current and next frames)
        /////
        /////  Get next non-root frame's frame_position_derivatives
        /////  based on these 2 intermediate variables:
        /////
        /////  1. current_frame_heavy_atom_X position
        /////  2. current_frame_quaternion ( q_w, q_i, q_j, q_k )
        /////
        /////  rotor X->Y here is denoted as ( rotor_x, rotor_y, rotor_z )
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

        );

        /////////////////////////////////////////////////////////
        /////
        /////  non-root frame version of func CoordDiffCore_GetRootFrameQuaternionDerivatives()
        /////
        /////////////////////////////////////////////////////////
        /////
        /////  calculate the next_frame_quaternion derivatives with respect to '6+K'
        /////
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

        );

    } // namespace differentiation

} // namespace pyvina

#endif // PYVINA_DIFFERENTIATION_COORD_DIFFERENTIATION_H_
