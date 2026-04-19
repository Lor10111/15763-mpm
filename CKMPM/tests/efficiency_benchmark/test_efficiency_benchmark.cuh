#pragma once

#include "mpm_test_base.h"
#include "mpm_material.cuh"
#include "mpm_model.h"
#include "mpm_domain.h"
#include "mpm_engine.cuh"
#include "data_type.cuh"
#include "mpm_config.h"

#include <array>
#include <vector>
#include <filesystem>
#include <string>
#include <string_view>

namespace mpm { namespace test {

// -----------------------------------------------------------------------
// Test 6: Efficiency profiling — shared benchmark scene
// Matches MLS-MPM test6_efficiency_profiling.py scene:
//   Elastic block center (0.5,0.5,0.7), size 0.40x0.40x0.40
//   Same material (FixedCorotated, E=1e5, nu=0.4, rho=1000)
//   profileMode=true → enables MPMBenchmark output (G2P2G timing)
// -----------------------------------------------------------------------
class MPMTestScene {
public:
    constexpr static std::string_view kTestName_ = "efficiency_benchmark";
    constexpr static float kDx_ = 1.0f / 256.f;
    constexpr static auto  kConstitutiveModel_ = MPMConstitutiveModel::kFixedCorotated;
    constexpr static float kE_  = 1e5f;
    constexpr static float kNu_ = 0.4f;
    constexpr static float kLambda_ = ComputeLameParameters<float>(kE_, kNu_)[0];
    constexpr static float kMu_     = ComputeLameParameters<float>(kE_, kNu_)[1];
    constexpr static auto  kMaterial_ = MPMMaterial<kConstitutiveModel_>{
        MPMMaterial<kConstitutiveModel_>::FixedCorotatedMaterialParameter{kLambda_, kMu_}};
    constexpr static float kRho_             = 1000.f;
    constexpr static float kParticlePerCell_ = 8.0f;
    constexpr static float kParticleVolume_  = kDx_*kDx_*kDx_ / kParticlePerCell_;
    constexpr static float kParticleMass_    = kParticleVolume_ * kRho_;
    constexpr static uint32_t kFps_          = 48;
    constexpr static float kCfl_             = 0.5f;
    constexpr static float kDtFactor_        = 1.0f;
    // Run 150 frames to match MLS-MPM timing test
    constexpr static float kTotalSimulatedTime_ = 150.f / 48.f;

    typedef MPMDomainRange<64, 64, 64> DomainRange_;
    typedef MPMDomainOffset<0, 0, 0>   DomainOffset_;
    typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
    typedef MPMGridConfig<Domain_> GridConfig_;

    class MPMEfficiencyConfig : public MPMConfigBase<MPMEfficiencyConfig> {
    public:
        friend class MPMConfigBase<MPMEfficiencyConfig>;
        constexpr MPMEfficiencyConfig() = default;
    protected:
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDxImpl()              const -> float    { return kDx_; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockVolumeImpl()     const -> uint32_t { return 64; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockSizeImpl()       const -> uint32_t { return 4; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerCellImpl()   const -> uint32_t { return 128; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBlockImpl()  const -> uint32_t { return 64*128; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBucketImpl() const -> uint32_t { return 32; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl()       const -> uint32_t { return 50000; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl()             const -> uint32_t { return kFps_; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl()              const -> float
            { return EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kE_, kNu_, kRho_, kCfl_); }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetCflImpl()             const -> float    { return kCfl_; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetTotalSimulatedFrameCountImpl() const -> uint32_t
            { return static_cast<uint32_t>(std::round(kTotalSimulatedTime_ * GetFpsImpl())); }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetGravityImpl()         const -> float    { return -9.8f; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMassClampImpl()       const -> float    { return 0.f; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistRigidParticleImpl()    const -> bool     { return false; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleCountImpl()    const -> uint32_t { return 0; }
        template<typename Scalar>
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleVelocityImpl() const -> Vector<Scalar, 3>
            { return Vector<Scalar, 3>{0.f, 0.f, 0.f}; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistIrregularBoundaryImpl() const -> bool { return false; }

        // Simple sticky floor only
        MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ProcessGridCellVelocityImpl(
            const Vector<int, 3>& cell, Vector<float, 3>& velocity) const -> void
        {
            float wy = (cell[1] + 0.5f) * kDx_;
            if (wy < 2.f * kDx_) { velocity[0]=0.f; velocity[1]=0.f; velocity[2]=0.f; }
        }

        MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto UpdateConfigImpl(float dt, int frame) -> void {}
    };
    typedef MPMEfficiencyConfig TestConfig_;
};

auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
{
    constexpr float kDx = MPMTestScene::kDx_;
    // Block center (128,179,128), half-size (51,51,51)  →  ~0.40 x 0.40 x 0.40
    // cy=179 → Y=0.70 (gravity is -Y in CKMPM)
    constexpr int cx=128, cy=179, cz=128, hx=51, hy=51, hz=51;
    std::vector<Vector<float,3>> position, velocity;
    for (int i=cx-hx; i<=cx+hx; ++i)
    for (int j=cy-hy; j<=cy+hy; ++j)
    for (int k=cz-hz; k<=cz+hz; ++k)
    for (int w=0; w<8; ++w) {
        int di=(w&4)>>2, dj=(w&2)>>1, dk=w&1;
        position.push_back(Vector<float,3>{(i+0.25f+di*0.5f)*kDx, (j+0.25f+dj*0.5f)*kDx, (k+0.25f+dk*0.5f)*kDx});
        velocity.push_back(Vector<float,3>{0.f, 0.f, 0.f});
        ++particleCount;
    }
    printf("efficiency_benchmark: %u particles\n", particleCount);
    return { MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{
        MPMTestScene::kParticleMass_, MPMTestScene::kParticleVolume_, position, velocity, MPMTestScene::kMaterial_}};
}

} } // namespace mpm::test
