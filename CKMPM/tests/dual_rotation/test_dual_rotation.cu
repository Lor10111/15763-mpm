#include "test_dual_rotation.cuh"
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

    // collectConservationMetric = true：
    // 输出 conservationMetric_model_0.bin（方块1）和 conservationMetric_model_1.bin（方块2）
    // 每帧 13 个 float：[frame | momentum[0..3] 各 xyz]
    // momentum[3] = Lagrangian 角动量，验证：model_0.z + model_1.z 应保持为 ~0
    simulator->Simulate(config, resultDirectory.c_str(), true, false, false);
}
