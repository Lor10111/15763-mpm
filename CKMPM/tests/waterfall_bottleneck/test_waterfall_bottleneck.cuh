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
// Test 4: Waterfall-to-bottleneck (fluid)
// Matches MLS-MPM test4_waterfall_bottleneck.py:
//   Fluid block: center (0.5,0.5,0.87), size 0.55x0.55x0.10
//   Bottleneck walls: X<0.38 and X>0.62 sticky for Z<0.35
// -----------------------------------------------------------------------
class MPMTestScene {
public:
    constexpr static std::string_view kTestName_ = "waterfall_bottleneck";
    constexpr static float kDx_ = 1.0f / 256.f;
    constexpr static auto  kConstitutiveModel_ = MPMConstitutiveModel::kFluid;
    constexpr static float kBulk_      = 10.f;
    constexpr static float kGamma_     = 7.15f;
    constexpr static float kViscosity_ = 0.1f;
    constexpr static auto  kMaterial_  = MPMMaterial<kConstitutiveModel_>{
        MPMMaterial<kConstitutiveModel_>::FluidMaterialParameter{kBulk_, kGamma_, kViscosity_}};
    constexpr static float kE_  = 5e3f;   // for timestep estimation only
    constexpr static float kNu_ = 0.4f;
    constexpr static float kRho_             = 1000.f;
    constexpr static float kParticlePerCell_ = 8.0f;
    constexpr static float kParticleVolume_  = kDx_*kDx_*kDx_ / kParticlePerCell_;
    constexpr static float kParticleMass_    = kParticleVolume_ * kRho_;
    constexpr static uint32_t kFps_          = 48;
    constexpr static float kCfl_             = 0.5f;
    constexpr static float kDtFactor_        = 1.0f;
    constexpr static float kTotalSimulatedTime_ = 150.f / 48.f;

    typedef MPMDomainRange<64, 64, 64> DomainRange_;
    typedef MPMDomainOffset<0, 0, 0>   DomainOffset_;
    typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
    typedef MPMGridConfig<Domain_> GridConfig_;

    class MPMWaterfallConfig : public MPMConfigBase<MPMWaterfallConfig> {
    public:
        friend class MPMConfigBase<MPMWaterfallConfig>;
        constexpr MPMWaterfallConfig() = default;
    protected:
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDxImpl()              const -> float    { return kDx_; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockVolumeImpl()     const -> uint32_t { return 64; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockSizeImpl()       const -> uint32_t { return 4; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerCellImpl()   const -> uint32_t { return 64; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBlockImpl()  const -> uint32_t { return 64*64; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBucketImpl() const -> uint32_t { return 32; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl()       const -> uint32_t { return 50000; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl()             const -> uint32_t { return kFps_; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl()              const -> float    { return 5e-5f; }
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

        // Bottleneck walls + floor
        MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ProcessGridCellVelocityImpl(
            const Vector<int, 3>& cell, Vector<float, 3>& velocity) const -> void
        {
            constexpr float dx      = kDx_;
            constexpr float neckXLo = 0.38f;
            constexpr float neckXHi = 0.62f;
            constexpr float yTop    = 0.35f;   // walls only below this height (gravity = -Y)

            float wx = (cell[0] + 0.5f) * dx;
            float wy = (cell[1] + 0.5f) * dx;

            // Sticky floor (Y direction)
            if (wy < 2.f * dx) { velocity[0]=0.f; velocity[1]=0.f; velocity[2]=0.f; return; }

            // Bottleneck walls (sticky): only active below yTop
            if (wy < yTop) {
                if (wx < neckXLo) { velocity[0] = fmaxf(velocity[0], 0.f); }  // left wall: cancel -X vel
                if (wx > neckXHi) { velocity[0] = fminf(velocity[0], 0.f); }  // right wall: cancel +X vel
            }
        }

        MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto UpdateConfigImpl(float dt, int frame) -> void {}
    };
    typedef MPMWaterfallConfig TestConfig_;
};

auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
{
    constexpr float kDx = MPMTestScene::kDx_;
    // Block center (128,222,128), half-size (70,13,70)
    // cy=222 → Y=0.87 (gravity is -Y); thin in Y (hy=13), wide in X/Z (hx=hz=70)
    constexpr int cx=128, cy=222, cz=128, hx=70, hy=13, hz=70;
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
    printf("waterfall_bottleneck: %u particles\n", particleCount);
    return { MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{
        MPMTestScene::kParticleMass_, MPMTestScene::kParticleVolume_, position, velocity, MPMTestScene::kMaterial_}};
}

} } // namespace mpm::test
