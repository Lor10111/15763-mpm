#include "test_dropping_block_fluid.cuh"
#include "mpm_engine.cuh"

int main()
{
    uint32_t particleCount = 0;
    auto modelList = mpm::test::SetupModel(particleCount);
    auto config = mpm::test::GetTestConfiguration<mpm::test::MPMTestScene>();

    std::filesystem::path resultDirectory = "result/" + mpm::test::GetTestName<mpm::test::MPMTestScene>() + "/";
    std::filesystem::create_directory(resultDirectory);

    auto simulator = mpm::test::GetSimulator<typename mpm::test::MPMTestScene>();
    simulator->Initialize(config);
    simulator->InitializeParticle(config, modelList);
    simulator->InitialSetup(config);

    // collectConservationMetric = true:
    // Writes conservationMetric_model_0.bin per frame (13 floats):
    // [frame | grid_lin×3 | grid_ang×3 | particle_lin×3 | particle_ang×3]
    // particle_lin (indices 7-9): Lagrangian linear momentum — tracks smooth flow through holes
    simulator->Simulate(config, resultDirectory.c_str(), true, false, false);
}
