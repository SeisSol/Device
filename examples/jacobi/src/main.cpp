#include "datatypes.hpp"
#include "helper.hpp"
#include "host/subroutines.hpp"
#include "matrix_manip.hpp"
#include "solvers.hpp"
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

int main(int argc, char *argv[]) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif
  WorkSpaceT ws{MPI_COMM_WORLD};
  int numRows{-1};
  SolverSettingsT settings;

  try {
    if (argc != 2) {
      throw std::runtime_error("please provide a single input file in yaml format");
    }

    YAML::Node params = YAML::LoadFile(argv[1]);
    Logger(ws, ws.rank) << (std::stringstream() << "Input parameters: \n" << params);

    numRows = params["num_rows"].as<int>();
    settings.eps = params["eps"].as<real>();
    settings.maxNumIters = params["max_num_iters"].as<unsigned>();
    settings.printInfoNumIters = params["iters_per_output"].as<unsigned>();
    std::string solverTypeStr = params["sovler_type"].as<std::string>();
    std::unordered_map<std::string, SolverType> solverTypeMap{{"cpu", SolverType::Cpu}, {"gpu", SolverType::Gpu}};

    if (solverTypeMap.find(solverTypeStr) != solverTypeMap.end()) {
      settings.solverType = solverTypeMap[solverTypeStr];
    } else {
      throw std::runtime_error("invalid solver type provided");
    }
  } catch (const std::runtime_error &error) {
    std::stringstream stream;
    stream << "Error: " << error.what();
    Logger(ws, ws.rank) << stream;
    return -1;
  } catch (const std::logic_error &error) {
    std::stringstream stream;
    stream << "Error: " << error.what();
    Logger(ws, ws.rank) << stream;
    return -1;
  }

  CpuMatrixDataT hostMatrix = init2DStencilMatrix(ws, numRows);

  VectorT knownSolutionVec(hostMatrix.info.numRows, 1.0);
  VectorT rhs(hostMatrix.info.numRows, 0.0);
  VectorT gpuRes(hostMatrix.info.numRows, 0.0);

  // compute RHS using the known solution
  host::multMatVec(hostMatrix, knownSolutionVec, rhs);
  VectorT unknown(hostMatrix.info.numRows, -10.0);

  // run solver
  auto start = std::chrono::steady_clock::now();
  switch (settings.solverType) {
  case SolverType::Cpu: {
    host::solver(settings, hostMatrix, rhs, unknown);
    break;
  }
  case SolverType::Gpu: {
    gpu::Solver solver;
    solver.run(settings, hostMatrix, rhs, unknown);
    break;
  }
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> time = end - start;

  // write output
  std::stringstream stream;
  if (unknown.size() < 20) {
    for (auto item : unknown) {
      stream << item << '\n';
    }
  }
  else {
    std::stringstream front, middle, back;
    constexpr int windowSize = 4;
    for (size_t i = 0; i < windowSize; ++i) {
      auto frontIndex = i;
      front << frontIndex << ") " << unknown[frontIndex] << '\n';

      auto middleIndex = ((unknown.size() / 2) - (windowSize / 2)) + i;
      middle << middleIndex << ") " << unknown[middleIndex] << '\n';

      auto backIndex = (unknown.size() - windowSize) + i;
      back << backIndex << ") " << unknown[backIndex] << '\n';
    }
    stream << front.str() << "...\n" << middle.str() << "...\n" << back.str();
  }

  Logger(ws, ws.rank) << (stream << "Elapsed Time: " << time.count() << " [sec]");
#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
