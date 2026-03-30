#include "test_translating_cube.cuh"// Includes scene definitions (MPMTestScene, SetupModel)
#include "mpm_engine.cuh"// MPMEngine main simulator class


int main()
{
    uint32_t particleCount = 0;
    auto modelList = mpm::test::SetupModel(particleCount);
    // mpm:: = library namespace; test:: = test utility sub-namespace
    // SetupModel generates particle positions + velocities on the CPU, returning a list of MPMModelVariant.
    auto config = mpm::test::GetTestConfiguration<mpm::test::MPMTestScene>();
    // Instantiates MPMTestScene::TestConfig_ (a CRTP configuration object).
    // This stores all compile-time parameters such as dx, dt, fps, gravity, etc.

    std::filesystem::path resultDirectory = "result/" + mpm::test::GetTestName<mpm::test::MPMTestScene>() + "/";
    std::filesystem::create_directory(resultDirectory);

    auto simulator = mpm::test::GetSimulator<typename mpm::test::MPMTestScene>();
    // Expands to: MPMEngine<MPMTestScene::GridConfig_>::GetInstance()
    // Returns a singleton pointer to the GPU simulator.
	simulator->Initialize(config);// Allocates grid, partition, and particle buffers on the GPU.
	simulator->InitializeParticle(config, modelList);// Copies particle data from CPU to GPU.
	simulator->InitialSetup(config);// Activates blocks containing particles and builds cell buckets.
	simulator->Simulate(config, resultDirectory.c_str(), true, false, false);
    // Param 3 = true: Writes conservationMetric_model_0.bin per frame (momentum conservation data).
    // Param 4 = false: Do not export raw binary positions.
    // Param 5 = false: Do not print GPU profiling/performance data.
    
}

