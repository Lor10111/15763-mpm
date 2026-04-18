#include "test_tasty_meal_water.cuh"
#include "mpm_engine.cuh"

int main()
{
    uint32_t particleCount = 0;
    auto modelList = mpm::test::SetupModel(particleCount);
    auto config    = mpm::test::GetTestConfiguration<mpm::test::MPMTestScene>();

    std::filesystem::path resultDirectory = "result/" + mpm::test::GetTestName<mpm::test::MPMTestScene>() + "/";
    std::filesystem::create_directory(resultDirectory);

    auto simulator = mpm::test::GetSimulator<typename mpm::test::MPMTestScene>();
    simulator->Initialize(config);
    simulator->InitializeParticle(config, modelList);
    simulator->InitialSetup(config);

    // collectConservationMetric = true:
    //   per-frame conservationMetric_model_{i}.bin (13 floats)
    //   [frame | grid_lin×3 | grid_ang×3 | particle_lin×3 | particle_ang×3]
    // → use these for momentum conservation metric (M1)
    simulator->Simulate(config, resultDirectory.c_str(), true, false, false);
}
