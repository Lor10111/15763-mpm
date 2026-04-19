#include "test_waterfall_bottleneck.cuh"
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
    simulator->Simulate(config, resultDir.c_str(), /*conservation=*/true, /*positions=*/false, /*profile=*/false);
    return 0;
}
