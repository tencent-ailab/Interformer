#include <chrono>
#include <iostream>
#include <iomanip>
#include <boost/program_options.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include "io_service_pool.hpp"
#include "safe_counter.hpp"
#include "random_forest.hpp"
#include "receptor.hpp"
#include "ligand.hpp"

int main(int argc, char* argv[])
{
	return 0;
	/////////
	// TODO: this func is going to be removed
	/////////


	// path receptor_path, ligand_path, out_path;
	// array<double, 3> center, size;
	// size_t seed, num_threads, num_trees, num_tasks, max_conformations;
	// double granularity;
	// bool score_only;

	// // Process program options.
	// try
	// {
	// 	// Initialize the default values of optional arguments.
	// 	const path default_out_path = ".";
	// 	const size_t default_seed = chrono::duration_cast<chrono::seconds>(chrono::system_clock::now().time_since_epoch()).count();
	// 	const size_t default_num_threads = boost::thread::hardware_concurrency();
	// 	const size_t default_num_trees = 500;
	// 	const size_t default_num_tasks = 64;
	// 	const size_t default_max_conformations = 9;
	// 	const double default_granularity = 0.125;

	// 	// Set up options description.
	// 	using namespace boost::program_options;
	// 	options_description input_options("input (required)");
	// 	input_options.add_options()
	// 		("receptor", value<path>(&receptor_path)->required(), "receptor in PDBQT format")
	// 		("ligand", value<path>(&ligand_path)->required(), "ligand or folder of ligands in PDBQT format")
	// 		("center_x", value<double>(&center[0])->required(), "x coordinate of the search space center")
	// 		("center_y", value<double>(&center[1])->required(), "y coordinate of the search space center")
	// 		("center_z", value<double>(&center[2])->required(), "z coordinate of the search space center")
	// 		("size_x", value<double>(&size[0])->required(), "size in the x dimension in Angstrom")
	// 		("size_y", value<double>(&size[1])->required(), "size in the y dimension in Angstrom")
	// 		("size_z", value<double>(&size[2])->required(), "size in the z dimension in Angstrom")
	// 		;
	// 	options_description output_options("output (optional)");
	// 	output_options.add_options()
	// 		("out", value<path>(&out_path)->default_value(default_out_path), "folder of predicted conformations in PDBQT format")
	// 		;
	// 	options_description miscellaneous_options("options (optional)");
	// 	miscellaneous_options.add_options()
	// 		("seed", value<size_t>(&seed)->default_value(default_seed), "explicit non-negative random seed")
	// 		("threads", value<size_t>(&num_threads)->default_value(default_num_threads), "number of worker threads to use")
	// 		("trees", value<size_t>(&num_trees)->default_value(default_num_trees), "number of decision trees in random forest")
	// 		("tasks", value<size_t>(&num_tasks)->default_value(default_num_tasks), "number of Monte Carlo tasks for global search")
	// 		("conformations", value<size_t>(&max_conformations)->default_value(default_max_conformations), "maximum number of binding conformations to write")
	// 		("granularity", value<double>(&granularity)->default_value(default_granularity), "density of probe atoms of grid maps")
	// 		("score_only", bool_switch(&score_only), "scoring without docking")
	// 		("help", "this help information")
	// 		("version", "version information")
	// 		("config", value<path>(), "configuration file to load options from")
	// 		;
	// 	options_description all_options;
	// 	all_options.add(input_options).add(output_options).add(miscellaneous_options);

	// 	// Parse command line arguments.
	// 	variables_map vm;
	// 	store(parse_command_line(argc, argv, all_options), vm);

	// 	// If no command line argument is supplied or help is requested, print the usage and exit.
	// 	if (argc == 1 || vm.count("help"))
	// 	{
	// 		cout << all_options;
	// 		return 0;
	// 	}

	// 	// If version is requested, print the version and exit.
	// 	if (vm.count("version"))
	// 	{
	// 		cout << "2.2.1" << endl;
	// 		return 0;
	// 	}

	// 	// If a configuration file is present, parse it.
	// 	if (vm.count("config"))
	// 	{
	// 		boost::filesystem::ifstream config_file(vm["config"].as<path>());
	// 		store(parse_config_file(config_file, all_options), vm);
	// 	}

	// 	// Notify the user of parsing errors, if any.
	// 	vm.notify();

	// 	// Validate receptor_path.
	// 	if (!exists(receptor_path))
	// 	{
	// 		cerr << "Option receptor " << receptor_path << " does not exist" << endl;
	// 		return 1;
	// 	}
	// 	if (!is_regular_file(receptor_path))
	// 	{
	// 		cerr << "Option receptor " << receptor_path << " is not a regular file" << endl;
	// 		return 1;
	// 	}

	// 	// Validate ligand_path.
	// 	if (!exists(ligand_path))
	// 	{
	// 		cerr << "Option ligand " << ligand_path << " does not exist" << endl;
	// 		return 1;
	// 	}

	// 	// Validate out_path.
	// 	if (exists(out_path))
	// 	{
	// 		if (!is_directory(out_path))
	// 		{
	// 			cerr << "Option out " << out_path << " is not a directory" << endl;
	// 			return 1;
	// 		}
	// 	}
	// 	else
	// 	{
	// 		if (!create_directories(out_path))
	// 		{
	// 			cerr << "Failed to create output folder " << out_path << endl;
	// 			return 1;
	// 		}
	// 	}

	// 	// Validate miscellaneous options.
	// 	if (!num_threads)
	// 	{
	// 		cerr << "Option threads must be 1 or greater" << endl;
	// 		return 1;
	// 	}
	// 	if (!num_tasks)
	// 	{
	// 		cerr << "Option tasks must be 1 or greater" << endl;
	// 		return 1;
	// 	}
	// 	if (!max_conformations)
	// 	{
	// 		cerr << "Option conformations must be 1 or greater" << endl;
	// 		return 1;
	// 	}
	// 	if (granularity <= 0)
	// 	{
	// 		cerr << "Option granularity must be positive" << endl;
	// 		return 1;
	// 	}
	// }
	// catch (const exception& e)
	// {
	// 	cerr << e.what() << endl;
	// 	return 1;
	// }

	// // Parse the receptor.
	// cout << "Parsing the receptor " << receptor_path << endl;
	// receptor rec(center, size, granularity);
    // rec.load_from_file(receptor_path.string());

	// // Reserve storage for result containers.
	// vector<vector<result>> result_containers(num_tasks);
	// for (auto& rc : result_containers)
	// {
	// 	rc.reserve(20);	// Maximum number of results obtained from a single Monte Carlo task.
	// }
	// vector<result> results;
	// results.reserve(max_conformations);

	// // Enumerate and sort input ligands.
	// cout << "Enumerating input ligands in " << ligand_path << endl;
	// vector<path> input_ligand_paths;
	// if (is_regular_file(ligand_path))
	// {
	// 	input_ligand_paths.push_back(ligand_path);
	// }
	// else
	// {
	// 	for (directory_iterator dir_iter(ligand_path), end_dir_iter; dir_iter != end_dir_iter; ++dir_iter)
	// 	{
	// 		// Filter files with .pdbqt and .PDBQT extensions.
	// 		const path input_ligand_path = dir_iter->path();
	// 		const auto ext = input_ligand_path.extension();
	// 		if (ext != ".pdbqt" && ext != ".PDBQT") continue;
	// 		input_ligand_paths.push_back(input_ligand_path);
	// 	}
	// }
	// const size_t num_input_ligands = input_ligand_paths.size();
	// cout << "Sorting " << num_input_ligands << " input ligands in alphabetical order" << endl;
	// sort(input_ligand_paths.begin(), input_ligand_paths.end());

	// // Initialize a Mersenne Twister random number generator.
	// cout << "Seeding a random number generator with " << seed << endl;
	// mt19937_64 rng(seed);

	// // Initialize an io service pool and create worker threads for later use.
	// cout << "Creating an io service pool of " << num_threads << " worker threads" << endl;
	// io_service_pool io(num_threads);
	// safe_counter<size_t> cnt;

	// // Precalculate the scoring function in parallel.
	// cout << "Calculating a scoring function of " << scoring_function::n << " atom types" << endl;
	// scoring_function sf;
	// cnt.init((sf.n + 1) * sf.n >> 1);
	// for (size_t t1 = 0; t1 < sf.n; ++t1)
	// for (size_t t0 = 0; t0 <=  t1; ++t0)
	// {
	// 	io.post([&, t0, t1]()
	// 	{
	// 		sf.precalculate(t0, t1);
	// 		cnt.increment();
	// 	});
	// }
	// cnt.wait();
	// sf.clear();

	// // Train RF-Score on the fly.
	// cout << "Training a random forest of " << num_trees << " trees with " << tree::nv << " variables and " << tree::ns << " samples" << endl;
	// forest f(num_trees, seed);
	// cnt.init(num_trees);
	// for (size_t i = 0; i < num_trees; ++i)
	// {
	// 	io.post([&, i]()
	// 	{
	// 		f[i].train(4, f.u01_s);
	// 		cnt.increment();
	// 	});
	// }
	// cnt.wait();
	// f.clear();

	// // Output headers to the standard output and the log file.
	// cout << "Creating grid maps of " << granularity << " A and running " << num_tasks << " Monte Carlo searches per ligand" << endl
	// 	<< "   Index             Ligand   nConfs   idock score (kcal/mol)   RF-Score (pKd)" << endl << setprecision(2);
	// cout.setf(ios::fixed, ios::floatfield);
	// boost::filesystem::ofstream log(out_path / "log.csv");
	// log.setf(ios::fixed, ios::floatfield);
	// log << "Ligand,nConfs,idock score (kcal/mol),RF-Score (pKd)" << endl << setprecision(2);

	// // Start to dock each input ligand.
	// size_t index = 0;
	// for (const auto& input_ligand_path : input_ligand_paths)
	// {
	// 	// Output the ligand file stem.
	// 	const string stem = input_ligand_path.stem().string();
	// 	cout << setw(8) << ++index << "   " << setw(16) << stem << "   " << flush;

	// 	// Check if the current ligand has already been docked.
	// 	size_t num_confs = 0;
	// 	double id_score = 0;
	// 	double rf_score = 0;
	// 	const path output_ligand_path = out_path / input_ligand_path.filename();
	// 	if (exists(output_ligand_path) && !equivalent(ligand_path, out_path))
	// 	{
	// 		// Extract idock score and RF-Score from output file.
	// 		string line;
	// 		for (boost::filesystem::ifstream ifs(output_ligand_path); getline(ifs, line);)
	// 		{
	// 			const string record = line.substr(0, 10);
	// 			if (record == "MODEL     ")
	// 			{
	// 				++num_confs;
	// 			}
	// 			else if (num_confs == 1 && record == "REMARK 921")
	// 			{
	// 				id_score = stod(line.substr(55, 8));
	// 			}
	// 			else if (num_confs == 1 && record == "REMARK 927")
	// 			{
	// 				rf_score = stod(line.substr(55, 8));
	// 			}
	// 		}
	// 	}
	// 	else
	// 	{
	// 		// Parse the ligand.
	// 		array<double, 3> origin;
	// 		ligand lig;
    //         origin = lig.load_from_file(input_ligand_path.string());

	// 		// Find atom types that are present in the current ligand but not present in the grid maps.
	// 		vector<size_t> xs;
	// 		for (size_t t = 0; t < sf.n; ++t)
	// 		{
	// 			if (lig.xs[t] && rec.maps[t].empty())
	// 			{
	// 				rec.maps[t].resize(rec.num_probes_product);
	// 				xs.push_back(t);
	// 			}
	// 		}

	// 		// Create grid maps on the fly if necessary.
	// 		if (xs.size())
	// 		{
	// 			// Precalculate p_offset.
	// 			rec.precalculate(xs);

	// 			// Populate the grid map task container.
	// 			cnt.init(rec.num_probes[2]);
	// 			for (size_t z = 0; z < rec.num_probes[2]; ++z)
	// 			{
	// 				io.post([&, z]()
	// 				{
	// 					rec.populate(xs, z, sf);
	// 					cnt.increment();
	// 				});
	// 			}
	// 			cnt.wait();
	// 		}

	// 		if (score_only)
	// 		{
	// 			num_confs = 1;
	// 			conformation c0(lig.num_active_torsions);
	// 			c0.position = origin;
	// 			double e0, f0;
	// 			change g0(0);
	// 			lig.evaluate(c0, sf, rec, -99, e0, f0, g0);
	// 			auto r0 = lig.compose_result(e0, f0, c0);
	// 			r0.e_nd = r0.f * lig.flexibility_penalty_factor;
	// 			r0.rf = lig.calculate_rf_score(r0, rec, f);
	// 			id_score = r0.e_nd;
	// 			rf_score = r0.rf;
	// 			lig.write_models(output_ligand_path, {{ move(r0) }}, rec);
	// 		}
	// 		else
	// 		{
	// 			// Run the Monte Carlo tasks.
	// 			cnt.init(num_tasks);
	// 			for (size_t i = 0; i < num_tasks; ++i)
	// 			{
	// 				assert(result_containers[i].empty());
	// 				const size_t s = rng();
	// 				io.post([&, i, s]()
	// 				{
	// 					lig.monte_carlo(result_containers[i], s, sf, rec);
	// 					cnt.increment();
	// 				});
	// 			}
	// 			cnt.wait();

	// 			// Merge results from all tasks into one single result container.
	// 			assert(results.empty());
	// 			const double required_square_error = static_cast<double>(4 * lig.num_heavy_atoms); // Ligands with RMSD < 2.0 will be clustered into the same cluster.
	// 			for (auto& result_container : result_containers)
	// 			{
	// 				for (auto& result : result_container)
	// 				{
	// 					result::push(results, move(result), required_square_error);
	// 				}
	// 				result_container.clear();
	// 			}

	// 			// If conformations are found, output them.
	// 			num_confs = results.size();
	// 			if (num_confs)
	// 			{
	// 				// Adjust free energy relative to the best conformation and flexibility.
	// 				const auto& best_result = results.front();
	// 				const double best_result_intra_e = best_result.e - best_result.f;
	// 				for (auto& result : results)
	// 				{
	// 					result.e_nd = (result.e - best_result_intra_e) * lig.flexibility_penalty_factor;
	// 					result.rf = lig.calculate_rf_score(result, rec, f);
	// 				}
	// 				id_score = best_result.e_nd;
	// 				rf_score = best_result.rf;

	// 				// Write models to file.
	// 				lig.write_models(output_ligand_path, results, rec);

	// 				// Clear the results of the current ligand.
	// 				results.clear();
	// 			}
	// 		}
	// 	}

	// 	// If output file or conformations are found, output the idock score and RF-Score.
	// 	cout << setw(6) << num_confs;
	// 	log << stem << ',' << num_confs;
	// 	if (num_confs)
	// 	{
	// 		cout << "   " << setw(22) << id_score << "   " << setw(14) << rf_score;
	// 		log << ',' << id_score << ',' << rf_score;
	// 	}
	// 	cout << endl;
	// 	log << '\n';

	// 	// Output to the log file in csv format. The log file can be sorted using: head -1 log.csv && tail -n +2 log.csv | awk -F, '{ printf "%s,%s\n", $2||0, $0 }' | sort -t, -k1nr -k3n | cut -d, -f2-
	// }

	// // Wait until the io service pool has finished all its tasks.
	// io.wait();
}
