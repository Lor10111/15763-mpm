#include "test_efficiency_benchmark.cuh"
#include "mpm_engine.cuh"

int main()
{
    uint32_t particleCount = 0;
    auto modelList = mpm::test::SetupModel(particleCount);
    auto config    = mpm::test::GetTestConfiguration<mpm::test::MPMTestScene>();

    std::filesystem::path resultDir = "result/" + std::string(mpm::test::GetTestName<mpm::test::MPMTestScene>()) + "/";
    std::filesystem::create_directories(resultDir);

    auto simulator = mpm::test::GetSimulator<typename mpm::test::MPMTestScene>();
    simulator->Initialize(config);
    simulator->InitializeParticle(config, modelList);
    simulator->InitialSetup(config);

    // profileMode=true → enables MPMBenchmark timing output (G2P2G, grid update etc.)
    // The benchmark results are printed to console by spdlog at end of simulation.
    // Redirect stdout to capture: .\mpm_test_efficiency_benchmark.exe > timing_ckmpm.txt
    simulator->Simulate(config, resultDir.c_str(), /*conservation=*/false, /*positions=*/false, /*profile=*/true);
    return 0;
}
