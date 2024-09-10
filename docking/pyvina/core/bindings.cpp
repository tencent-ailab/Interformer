#include "ligand.hpp"
#include "receptor.hpp"
#include "scoring_function.hpp"
#include "result.hpp"

#include "tests/tensor_tests.h"
#include <torch/extension.h>

// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

// #include <torch/torch.h>

#include <omp.h>
#include <chrono>

namespace py = pybind11;

void ShowTorchTensors()
{
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
}

// Utility function for grid map calculation.
void receptor_create_grid_map(ligand &lig, receptor &rec, scoring_function &sf)
{

    if (rec.has_precalculated)
    {
        return;
    }

    // Find atom types that are present in the current ligand but not present in the grid maps.
    vector<size_t> xs;
    for (size_t t = 0; t < sf.n; ++t)
    {
        if (lig.xs[t] && rec.maps[t].empty())
        {
            rec.maps[t].resize(rec.num_probes_product);
            xs.push_back(t);
        }
    }
    // Create grid maps on the fly if necessary.
    if (xs.size())
    {
        // Precalculate p_offset.
        rec.precalculate(xs);
        // Populate the grid map task container.
#pragma omp parallel for
        for (size_t z = 0; z < rec.num_probes[2]; ++z)
        {
            rec.populate(xs, z, sf);
        }
    }

    rec.has_precalculated = true;
}

std::vector<std::vector<double>> receptor_get_maps(receptor &rec)
{
    return rec.maps;
}

std::vector<result> run_monte_carlo(std::string output_ligand_path, ligand &lig, receptor &rec, scoring_function &sf, int num_tasks, int max_conformations)
{
    const size_t default_seed = chrono::duration_cast<chrono::seconds>(chrono::system_clock::now().time_since_epoch()).count();

    std::vector<size_t> seeds;
    std::vector<std::vector<result>> result_containers(num_tasks);
    vector<result> results;
    results.reserve(max_conformations);

    for (auto &rc : result_containers)
    {
        rc.reserve(20); // Maximum number of results obtained from a single Monte Carlo task.
    }

    seeds.resize(num_tasks);

    mt19937_64 rng(default_seed);

    for (int i = 0; i < num_tasks; ++i)
    {
        seeds[i] = rng();
    }

#pragma omp parallel for
    for (int i = 0; i < num_tasks; ++i)
    {
        lig.monte_carlo(result_containers[i], seeds[i], sf, rec);
    }

    assert(results.empty());
    const double required_square_error = static_cast<double>(4 * lig.num_heavy_atoms); // Ligands with RMSD < 2.0 will be clustered into the same cluster.
    for (auto &result_container : result_containers)
    {
        for (auto &result : result_container)
        {
            result::push(results, move(result), required_square_error);
        }
        result_container.clear();
    }

    size_t num_confs = 0;
    // Write models to file.
    const auto &best_result = results.front();
    const double best_result_intra_e = best_result.e - best_result.f;
    for (auto &result : results)
    {
        result.e_nd = (result.e - best_result_intra_e) * lig.flexibility_penalty_factor;
    }
    return results;
}

void write_models(const std::string path, const std::vector<result> results, ligand &lig, receptor &rec)
{
    lig.write_models(path, results, rec);
}

double get_id_score(std::vector<result> results)
{
    const auto &best_result = results.front();
    return best_result.e_nd;
}

//std::vector<result> rescore_with_rf(std::vector<result> results, random_forest& rf) {
//    // If conformations are found, output them.
//    num_confs = results.size();
//    double id_score = 0;
//    double rf_score = 0;
//    if (num_confs)
//    {
//        // Adjust free energy relative to the best conformation and flexibility.
//        const auto& best_result = results.front();
//        const double best_result_intra_e = best_result.e - best_result.f;
//        for (auto& result : results)
//        {
//            result.e_nd = (result.e - best_result_intra_e) * lig.flexibility_penalty_factor;
//            result.rf = lig.calculate_rf_score(result, rec, f);
//        }
//        id_score = best_result.e_nd;
//        rf_score = best_result.rf;
//    }
//    return results;
//}

#include "quaternion/quaternion_torch.h"

#include "evaluator/eval_vina.h"
#include "evaluator/evaluator_vina_vanilla.h"
#include "evaluator/evaluator_vina_formula_derivative.h"
// #include "evaluator/core_evaluator_distancemap_grid.h"

#include "pyvina/core/evaluators/core_evaluator_base.h"
#include "pyvina/core/evaluators/grid_4d/core_evaluator_distancemap.h"
#include "pyvina/core/evaluators/grid_4d/core_evaluator_grid_4d.h"
#include "pyvina/core/evaluators/grid_4d/core_evaluator_normalscore.h"
#include "pyvina/core/evaluators/grid_4d/core_evaluator_vinascore.h"
#include "pyvina/core/evaluators/grid_4d/core_grid_4d.h"

#include "pyvina/core/minimizers/core_minimizer_base.h"
#include "pyvina/core/minimizers/core_minimizer_bfgs.h"

#include "pyvina/core/samplers/core_sampler_base.h"
#include "pyvina/core/samplers/core_sampler_monte_carlo.h"

#include"pyvina/core/wrappers_for_random/common.h"

#include "minimizer/bfgs_torch.h"
#include "minimizer/bfgs_formula_derivative.h"

#include "scoring_function/x_score_scoring_function.h"

#include "differentiation/coord_differentiation.h"

PYBIND11_MODULE(pyvina_core, m)
{
    m.def("receptor_create_grid_map", &receptor_create_grid_map, "Create receptor grid map wrt to the ligand.");
    m.def("receptor_get_maps", &receptor_get_maps, "Copy maps from an receptor instance");
    m.def("run_monte_carlo", &run_monte_carlo, "Run monte carlo simulation");
    m.def("get_id_score", &get_id_score, "Get vina score from results");
    m.def("write_models", &write_models, "Write models to file");

    // For test purposes only:
    m.def("ShowTorchTensors", &ShowTorchTensors);
    m.def("add_1", &AddOne);
    m.def("quaternion_mul_cpp", &pyvina::QuaternionMul);

    {
      py::module_ m2 = m.def_submodule("wrappers_for_random", "a submodule for bindings regarding random");
      m2.def("SetGeneratorsForThreadsGivenSeeds",
             &pyvina::core::wrappers_for_random::SetGeneratorsForThreadsGivenSeeds);
    }

    ///////////////////////////////////////
    /////
    /////   Differentiation related:
    /////   (for test purposes only)
    /////
    ///////////////////////////////////////
    m.def("Differentiation_CoordDiffCore_GetHeavyAtomCoordDerivatives",
          &pyvina::differentiation::CoordDiffCore_GetHeavyAtomCoordDerivatives);

    m.def("Differentiation_CoordDiffCore_GetRootFramePositionDerivatives",
          &pyvina::differentiation::CoordDiffCore_GetRootFramePositionDerivatives);
    m.def("Differentiation_CoordDiffCore_GetRootFrameQuaternionDerivatives",
          &pyvina::differentiation::CoordDiffCore_GetRootFrameQuaternionDerivatives);

    m.def("Differentiation_CoordDiffCore_GetNonRootFramePositionDerivatives",
          &pyvina::differentiation::CoordDiffCore_GetNonRootFramePositionDerivatives);
    m.def("Differentiation_CoordDiffCore_GetNonRootFrameQuaternionDerivatives",
          &pyvina::differentiation::CoordDiffCore_GetNonRootFrameQuaternionDerivatives);

    ///////////////////////////////////////
    /////
    /////   Scoring Function related:
    /////
    ///////////////////////////////////////
    py::class_<pyvina::scoring_function::XScoreScoringFunction>(m, "XScoreScoringFunctionCpp")
        .def(py::init<int, int, double>())
        .def("Score", &pyvina::scoring_function::XScoreScoringFunction::Score);

    /////////////////////////////////////////////////////////////
    /////
    /////   Evaluator-related:
    /////
    /////////////////////////////////////////////////////////////
    // m.def("EvalVina_GetEnergyFromConformation", &pyvina::EvalVina_GetEnergyFromConformation);
    m.def("eval_vina_get_energy_from_conformation_cppbind_", &pyvina::eval_vina_get_energy_from_conformation);
    
    m.def("MinimizerBfgsTorchCore", &pyvina::minimizer::BfgsTorchCore);
    m.def("Minimizer_BfgsFormulaDerivativeCore", &pyvina::minimizer::BfgsFormulaDerivativeCore);

    m.def("EvaluatorVinaVanillaCore_ConformationToEnergy", &pyvina::evaluator::VinaVanillaCore_ConformationToEnergy);
    m.def("Evaluator_VinaFormulaDerivativeCore_ConformationToEnergy",
          &pyvina::evaluator::VinaFormulaDerivativeCore_ConformationToEnergy);

    {
      ///////////////////////////////////////
      ///// submodule [ evaluators ]
      ///////////////////////////////////////
      py::module_ m2 = m.def_submodule("evaluators", "a submodule for bindings regarding evaluators");

      {
        // for debug only
        using pyvina::core::evaluators::CoreEvaluatorBase;
        py::class_<CoreEvaluatorBase>(m2, "CoreEvaluatorBase");
      }

      {
        // for debug only
        using pyvina::core::evaluators::grid_4d::CoreEvaluatorGrid4D;
        py::class_<CoreEvaluatorGrid4D>(m2, "CoreEvaluatorGrid4D");
      }

      {
        using pyvina::core::evaluators::CoreEvaluatorBase;
        using pyvina::core::evaluators::grid_4d::CoreEvaluatorDistancemap;
        py::class_<CoreEvaluatorDistancemap, CoreEvaluatorBase>(m2, "CoreEvaluatorDistancemap")
            .def(py::init<

                 // ligand
                 const ligand &,
                 // inter-molecular distance map
                 std::vector<std::vector<double>>,
                 // pocket
                 std::array<double, 3>, std::array<double, 3>,
                 // resolution
                 int,
                 // vdwsum[u_index_ligand][v_index_receptor] => u_vdwradius + v_vdwradius
                 std::vector<std::vector<double>>,
                 // e.g. receptor atom positions
                 std::vector<std::array<double, 3>>

                 >())
            .def("evaluate", &CoreEvaluatorDistancemap::evaluate)
            .def("get_evaluatorname", &CoreEvaluatorDistancemap::get_evaluatorname)
            .def("precalculate_core_grid_4d_", &CoreEvaluatorDistancemap::precalculate_core_grid_4d_)
            .def_readonly("core_grid_4d_", &CoreEvaluatorDistancemap::core_grid_4d_);
      }

      {
        using pyvina::core::evaluators::CoreEvaluatorBase;
        using pyvina::core::evaluators::grid_4d::CoreEvaluatorNormalscore;
        py::class_<CoreEvaluatorNormalscore, CoreEvaluatorBase>(m2, "CoreEvaluatorNormalscore")
            .def(py::init<

                 // ligand
                 const ligand &,
                 // normal distribution related
                 std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>,
                 std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>,
                 // e.g. receptor atom positions
                 std::vector<std::array<double, 3>>,
                 // pocket
                 std::array<double, 3>, std::array<double, 3>,
                 // resolution
                 int

                 >())
            .def("evaluate", &CoreEvaluatorNormalscore::evaluate)
            .def("evaluate_2debug", &CoreEvaluatorNormalscore::evaluate_2debug)
            .def("evaluate_no_grid", &CoreEvaluatorNormalscore::evaluate_no_grid)
            .def("get_str_address", &CoreEvaluatorNormalscore::get_str_address)
            .def("get_evaluatorname", &CoreEvaluatorNormalscore::get_evaluatorname)
            .def("precalculate_core_grid_4d_", &CoreEvaluatorNormalscore::precalculate_core_grid_4d_)
            .def("calculate_no_grid_loss_inter_for_1_heavy_atom",
                 &CoreEvaluatorNormalscore::calculate_no_grid_loss_inter_for_1_heavy_atom)
            .def("get_weight_intra", &CoreEvaluatorNormalscore::get_weight_intra)
            .def("set_weight_intra", &CoreEvaluatorNormalscore::set_weight_intra)
            .def("get_weight_collision_inter", &CoreEvaluatorNormalscore::get_weight_collision_inter)
            .def("set_weight_collision_inter", &CoreEvaluatorNormalscore::set_weight_collision_inter)
            .def("SetOption", &CoreEvaluatorNormalscore::SetOption)
            .def("ResetOption", &CoreEvaluatorNormalscore::ResetOption)
            .def("GetOption", &CoreEvaluatorNormalscore::GetOption)
            .def("evaluate_nogrid_given_weights", &CoreEvaluatorNormalscore::evaluate_nogrid_given_weights)
            .def_readonly("core_grid_4d_", &CoreEvaluatorNormalscore::core_grid_4d_);
      }

      {
        using pyvina::core::evaluators::CoreEvaluatorBase;
        using pyvina::core::evaluators::grid_4d::CoreEvaluatorVinascore;
        py::class_<CoreEvaluatorVinascore, CoreEvaluatorBase>(m2, "CoreEvaluatorVinascore")
            .def(py::init<

                 // ligand
                 const ligand &,
                 // receptor
                 const receptor &,
                 // pocket
                 std::array<double, 3>, std::array<double, 3>,
                 // resolution
                 int,
                 // vdwsum[u_index_ligand][v_index_receptor] => u_vdwradius + v_vdwradius
                 std::vector<std::vector<double>>,
                 // e.g. receptor atom positions
                 std::vector<std::array<double, 3>>

                 >())
            .def("evaluate", &CoreEvaluatorVinascore::evaluate)
            .def("precalculate_core_grid_4d_", &CoreEvaluatorVinascore::precalculate_core_grid_4d_);
      }

      {
        using pyvina::core::evaluators::grid_4d::CoreGrid4D;
        py::class_<CoreGrid4D>(m2, "CoreGrid4D")
            .def_readonly("corner_min_", &CoreGrid4D::corner_min_)
            .def_readonly("corner_max_", &CoreGrid4D::corner_max_)
            .def_readonly("double_shape3d_", &CoreGrid4D::double_shape3d_)
            .def_readonly("int_shape3d_", &CoreGrid4D::int_shape3d_)
            .def_readonly("int_shape3d_with_margin_", &CoreGrid4D::int_shape3d_with_margin_)
            .def_readonly("reciprocal_resolution_", &CoreGrid4D::reciprocal_resolution_)
            .def_readonly("resolution_", &CoreGrid4D::resolution_)
            .def_readonly("value_", &CoreGrid4D::value_);
      }
      ///////////////////////////////////////
      ///// submodule [ evaluators ]
      ///////////////////////////////////////
    }

    {
      ///////////////////////////////////////
      ///// submodule [ minimizers ]
      ///////////////////////////////////////
      py::module_ m2 = m.def_submodule("minimizers", "a submodule for bindings regarding minimizers");

      {
        using pyvina::core::minimizers::CoreMinimizerBase;
        py::class_<CoreMinimizerBase>(m2, "CoreMinimizerBase");
      }

      {
        using pyvina::core::evaluators::CoreEvaluatorBase;
        using pyvina::core::minimizers::CoreMinimizerBase;
        using pyvina::core::minimizers::CoreMinimizerBfgs;

        py::class_<CoreMinimizerBfgs, CoreMinimizerBase>(m2, "CoreMinimizerBfgs")
            .def(py::init<

                 const CoreEvaluatorBase &

                 >())
            .def("MinimizeGivenEvaluator", &CoreMinimizerBfgs::MinimizeGivenEvaluator)
            .def("GetStrAddressOfEvaluator", &CoreMinimizerBfgs::GetStrAddressOfEvaluator)
            .def("Minimize", &CoreMinimizerBfgs::Minimize);
      }
      ///////////////////////////////////////
      ///// submodule [ minimizers ]
      ///////////////////////////////////////
    }

    {
      ///////////////////////////////////////
      ///// submodule [ minimizers ]
      ///////////////////////////////////////
      py::module_ m2 = m.def_submodule("samplers", "a submodule for bindings regarding samplers");

      {
        using pyvina::core::samplers::CoreSamplerBase;
        py::class_<CoreSamplerBase>(m2, "CoreSamplerBase");
      }

      {
        using pyvina::core::minimizers::CoreMinimizerBase;
        using pyvina::core::samplers::CoreSamplerBase;
        using pyvina::core::samplers::CoreSamplerMonteCarlo;

        py::class_<CoreSamplerMonteCarlo, CoreSamplerBase>(m2, "CoreSamplerMonteCarlo")
            .def(py::init<

                 const CoreMinimizerBase &

                 >())
            .def("Sample", &CoreSamplerMonteCarlo::Sample);
      }
      ///////////////////////////////////////
      ///// submodule [ minimizers ]
      ///////////////////////////////////////
    }

    // py::class_<pyvina::core_evaluators::CoreEvaluatorDistancemapGrid>(m, "CoreEvaluatorDistancemapGrid")
    //     .def(py::init<

    //          // ligand
    //          ligand,

    //          // inter-molecular distance map
    //          std::vector<std::vector<double>>, std::vector<std::array<double, 3>>, std::vector<std::vector<double>>,

    //          // pocket
    //          std::array<double, 3>, std::array<double, 3>,

    //          // resolution
    //          int

    //          >())
    //     .def("evaluate_grid_based", &pyvina::core_evaluators::CoreEvaluatorDistancemapGrid::evaluate_grid_based)
    //     .def_readonly("inter_distancemap_", &pyvina::core_evaluators::CoreEvaluatorDistancemapGrid::inter_distancemap_)
    //     .def_readonly("list_inter_coordinates_",
    //                   &pyvina::core_evaluators::CoreEvaluatorDistancemapGrid::list_inter_coordinates_)
    //     .def_readonly("inter_weightmap_", &pyvina::core_evaluators::CoreEvaluatorDistancemapGrid::inter_weightmap_)
    //     .def_readonly("core_grid_4d_", &pyvina::core_evaluators::CoreEvaluatorDistancemapGrid::core_grid_4d_);

    py::class_<atom>(m, "AtomCpp")
        .def(py::init<const string &>())
        .def_readwrite("coord", &atom::coord)
        .def_readwrite("name", &atom::name)
        .def_readwrite("element", &atom::rf)
        .def_readwrite("xs", &atom::xs)
        .def_readonly("partial_charge", &atom::partial_charge);

    py::class_<frame>(m, "FrameCpp")
        .def(py::init<const size_t, const size_t, const size_t, const size_t, const size_t, const size_t>())
        .def_readwrite("parent", &frame::parent)
        .def_readwrite("rotor_x_idx", &frame::rotorXidx)
        .def_readwrite("rotor_y_idx", &frame::rotorYidx)
        .def_readwrite("ha_begin", &frame::habegin)
        .def_readwrite("ha_end", &frame::haend)
        .def_readwrite("hy_begin", &frame::hybegin)
        .def_readwrite("hy_end", &frame::hyend)
        .def_readwrite("active", &frame::active)
        .def_readwrite("parent_rotor_y_to_current_rotor_y", &frame::parent_rotorY_to_current_rotorY)
        .def_readwrite("parent_rotor_x_to_current_rotor_y", &frame::parent_rotorX_to_current_rotorY);

    py::class_<interacting_pair>(m, "interacting_pair", py::dynamic_attr())
        .def_readonly("i0", &interacting_pair::i0)
        .def_readonly("i1", &interacting_pair::i1);

    // py::class_<ligand>(m, "ligand", py::dynamic_attr())
    py::class_<ligand>(m, "LigandBaseCpp", py::dynamic_attr())
        .def(py::init<>())
        .def("load_from_file_cppbind_", &ligand::load_from_file)
        // .def("GetCoordsFromConformation", &ligand::GetCoordsFromConformation)
        .def("get_coords_from_conformation_cppbind_", &ligand::get_coords_from_conformation)
        .def("VanillaCore_ConformationToCoords", &ligand::VanillaCore_ConformationToCoords)
        .def("FormulaDerivativeCore_ConformationToCoords", &ligand::FormulaDerivativeCore_ConformationToCoords)
        .def("GetPoseInitial", &ligand::GetPoseInitial)
        .def("GetPoseRandomInBoundingbox", &ligand::GetPoseRandomInBoundingbox)
        .def("GetPoseRandomNext", &ligand::GetPoseRandomNext)
        .def("PickFirstKUniquePoses", &ligand::PickFirstKUniquePoses)
        .def_readonly("pdbqt_file_path_", &ligand::pdbqt_file_path_)
        .def_readonly("interacting_pairs_cppbind_", &ligand::interacting_pairs)
        .def_readonly("lines_cppbind_", &ligand::lines)
        .def_readonly("heavy_atoms_cppbind_", &ligand::heavy_atoms)
        .def_readonly("hydrogens_cppbind_", &ligand::hydrogens)
        .def_readonly("frames_cppbind_", &ligand::frames)
        .def_readonly("num_heavy_atoms", &ligand::num_heavy_atoms)
        .def_readonly("num_active_torsions", &ligand::num_active_torsions)
        .def_readonly("num_frames", &ligand::num_frames)
        .def_readonly("num_torsions", &ligand::num_torsions)
        .def_readonly("origin", &ligand::heavy_atoms_origin);

    // py::class_<receptor>(m, "receptor", py::dynamic_attr())
    py::class_<receptor>(m, "ReceptorBaseCpp", py::dynamic_attr())
        .def(py::init<const array<double, 3>, const array<double, 3>, const double>())
        .def("load_from_file", &receptor::load_from_file)
        .def("create_grid_map", &receptor::create_grid_map)
        .def_readonly("center", &receptor::center)
        .def_readonly("corner0", &receptor::corner0)
        .def_readonly("corner1", &receptor::corner1)
        .def_readonly("granularity", &receptor::granularity)
        .def_readonly("granularity_inverse", &receptor::granularity_inverse)
        .def_readonly("num_probes", &receptor::num_probes)
        .def_readonly("has_precalculated", &receptor::has_precalculated)
        .def_readonly("maps", &receptor::maps)
        .def_readonly("pdbqt_file_path_", &receptor::pdbqt_file_path_)
        .def_readwrite("atoms", &receptor::atoms);

    py::class_<scoring_function>(m, "scoring_function", py::dynamic_attr())
        .def(py::init<>())
        .def("create_grid_map", &scoring_function::create_grid_map)
        .def("score", &scoring_function::score_pybind11)
        .def_readonly("num_samples_per_angstrom", &scoring_function::num_samples_per_angstrom)
        .def_readonly("num_samples_within_cutoff", &scoring_function::num_samples_within_cutoff)
        .def_readonly("cutoff_angstrom", &scoring_function::cutoff_angstrom)
        .def_readonly("cutoff_sqr_angstrom2", &scoring_function::cutoff_sqr_angstrom2)
        .def_readonly("e", &scoring_function::e);
    // .def( "score" , static_cast<double (scoring_function::*)( const size_t , const size_t , const double )>(&scoring_function::score) );

    // How to bind a oveloaded function? Checked the document but have not resolved this issue:
    // https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods

    py::class_<result>(m, "result")
        .def(py::init<const double, const double, vector<array<double, 3>> &&, vector<array<double, 3>> &&>())
        .def_readwrite("free_energy", &result::e)
        .def_readwrite("inter_mol", &result::f)
        .def_readwrite("free_energy_norm", &result::e_nd);
}
