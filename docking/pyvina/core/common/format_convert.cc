
#include <iostream>
#include <fstream>
// #include <openbabel/obconversion.h>

#include "format_convert.h"

namespace pyvina
{

    namespace common
    {

        /*

        std::string ConvertPdbqtToAsTempFilePath(
            std::string file_path,
            std::string file_type_return,
            bool show_log)
        {
            std::ifstream fin(file_path);
            std::string tmp_file_path = file_path + ".pyvina_tmp." + file_type_return;
            std::ofstream fout(tmp_file_path);

            system(
                ("/usr/bin/obabel -i PDBQT " + file_path + " -O " + tmp_file_path).c_str()
                );
            
            ///// FIXME: core dump issues from openbabel library

            // OpenBabel::OBConversion conv(&fin, &fout);
            // if (!conv.SetInAndOutFormats("PDBQT", file_type_return.c_str()))
            // {
            //     if (show_log)
            //     {
            //         std::cout
            //             << " :: ERROR :: Formats not available: " << file_path << std::endl
            //             << " :: ERROR :: From PDBQT to " << file_type_return << std::endl;
            //     }
            //     return "";
            // }

            // int n = conv.Convert();
            // if (show_log)
            // {
            //     std::cout
            //         << " :: In " << file_path << std::endl
            //         << " :: " << n << " molecules converted from PDBQT to " << file_type_return << std::endl;
            // }

            return tmp_file_path;
        }

        // example of format convert from:
        // https://openbabel.org/docs/dev/UseTheLibrary/CppExamples.html

        using namespace std;

        int FormatConvertExample(int argc, char **argv)
        {
            if (argc < 3)
            {
                cout << "Usage: ProgrameName InputFileName OutputFileName\n";
                return 1;
            }

            ifstream ifs(argv[1]);
            if (!ifs)
            {
                cout << "Cannot open input file\n";
                return 1;
            }
            ofstream ofs(argv[2]);
            if (!ofs)
            {
                cout << "Cannot open output file\n";
                return 1;
            }
            OpenBabel::OBConversion conv(&ifs, &ofs);
            if (!conv.SetInAndOutFormats("PDBQT", "MOL2"))
            {
                cout << "Formats not available\n";
                return 1;
            }
            int n = conv.Convert();
            cout << n << " molecules converted\n";

            return 0;
        }

        */

    } // namespace common

} // namespace pyvina
