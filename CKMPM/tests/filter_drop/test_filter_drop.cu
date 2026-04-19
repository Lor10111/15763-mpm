#include "test_filter_drop.cuh"
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

    // collectConservationMetric=true  → writes conservationMetric_model_0.bin
    // exportParticlePositionAsBinary=true → writes per-frame position files
    // profileMode=false → normal render output
    simulator->Simulate(config, resultDir.c_str(), /*conservation=*/true, /*positions=*/true, /*profile=*/false);
    return 0;
}
