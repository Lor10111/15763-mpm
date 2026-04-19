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
#include <fstream>
#include <string>
#include <string_view>

namespace mpm { namespace test {

// -----------------------------------------------------------------------
// Test 2: Material block dropping through 3 horizontal cylinder filters
// Matches MLS-MPM test2_filter_drop.py scene:
//   Block center (0.5, 0.5, 0.80), size ~0.40x0.40x0.14
//   3 cylinders Y-axis aligned at X=0.20,0.50,0.80, Z=0.50, radius=0.06
// -----------------------------------------------------------------------
class MPMTestScene {
public:
    constexpr static std::string_view kTestName_ = "filter_drop";
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
    constexpr static float kParticleVolume_  = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
    constexpr static float kParticleMass_    = kParticleVolume_ * kRho_;
    constexpr static uint32_t kFps_          = 48;
    constexpr static float kCfl_             = 0.5f;
    constexpr static float kDtFactor_        = 1.0f;
    constexpr static float kTotalSimulatedTime_ = 150.f / 48.f;

    typedef MPMDomainRange<64, 64, 64> DomainRange_;
    typedef MPMDomainOffset<0, 0, 0>   DomainOffset_;
    typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
    typedef MPMGridConfig<Domain_> GridConfig_;

    class MPMFilterDropConfig : public MPMConfigBase<MPMFilterDropConfig> {
    public:
        friend class MPMConfigBase<MPMFilterDropConfig>;
        constexpr MPMFilterDropConfig() = default;
    protected:
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDxImpl()              const -> float    { return kDx_; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockVolumeImpl()     const -> uint32_t { return 64; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockSizeImpl()       const -> uint32_t { return 4; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerCellImpl()   const -> uint32_t { return 64; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBlockImpl()  const -> uint32_t { return 64 * 64; }
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

        // Cylinder + floor collider applied every substep on each grid cell
        // Gravity is -Y in CKMPM; cylinders are Z-aligned (horizontal), distance in XY plane
        MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ProcessGridCellVelocityImpl(
            const Vector<int, 3>& cell, Vector<float, 3>& velocity) const -> void
        {
            constexpr float dx     = kDx_;
            constexpr float radius = 0.06f;
            constexpr float cylY   = 0.50f;
            float wx = (cell[0] + 0.5f) * dx;
            float wy = (cell[1] + 0.5f) * dx;

            // Sticky floor (Y direction)
            if (wy < 2.f * dx) { velocity[0]=0.f; velocity[1]=0.f; velocity[2]=0.f; return; }

            // 3 Z-aligned cylinders at X = 0.20, 0.50, 0.80, Y = 0.50
            const float cxArr[3] = {0.20f, 0.50f, 0.80f};
            for (int c = 0; c < 3; ++c) {
                float ex    = wx - cxArr[c];
                float ey    = wy - cylY;
                float dist2 = ex*ex + ey*ey;
                if (dist2 < radius*radius && dist2 > 1e-12f) {
                    float dist = sqrtf(dist2);
                    float nx = ex/dist, ny = ey/dist;
                    float dot = velocity[0]*nx + velocity[1]*ny;
                    if (dot < 0.f) { velocity[0] -= dot*nx; velocity[1] -= dot*ny; }
                }
            }
        }

        MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto UpdateConfigImpl(float dt, int frame) -> void {}
    };
    typedef MPMFilterDropConfig TestConfig_;
};

auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
{
    constexpr float kDx = MPMTestScene::kDx_;
    // Block center (128,205,128), half-size (51,18,51) in grid indices
    // cy=205 → Y=0.80 (gravity is -Y); thin in Y (hy=18), wide in X/Z (hx=hz=51)
    constexpr int cx=128, cy=205, cz=128, hx=51, hy=18, hz=51;
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
    printf("filter_drop: %u particles\n", particleCount);
    return { MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{
        MPMTestScene::kParticleMass_, MPMTestScene::kParticleVolume_, position, velocity, MPMTestScene::kMaterial_}};
}

} } // namespace mpm::test
