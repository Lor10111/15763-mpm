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

// ─────────────────────────────────────────────────────────────────────────────
// Colliding Cubes – shared physical parameters
//
//  Two identical elastic cubes are launched toward each other along the x-axis.
//  No gravity, no external boundary constraints, no rigid bodies.
//
//  Shared across CKMPM / MLS-MPM / Basic-MPM-3D:
//    E          = 1e6 Pa           (Fixed Corotated)
//    nu         = 0.4
//    rho        = 1000 kg/m³
//    v0         = ±(0.1, 0, 0) m/s
//    ppc        = 8  (2×2×2 sub-cell)
//    CFL        = 0.5
//    fps        = 48
//    total_time = 3.0 s
//
//  Grid resolution:
//    CKMPM:       dx = 1/64
//    MLS-MPM:     dx = 1/64
//    Basic-MPM:   dx = 1/64
//
//  Cube geometry (in cell coordinates, same loop structure in all implementations):
//    Loop i,j,k ∈ [−4, +4]  →  9 cells per side
//    2×2×2 sub-cell offsets  →  9³ × 8 = 5 832 particles per cube
//    Cube 1 centre: cell (16, 32, 32) = (0.25, 0.50, 0.50) m
//    Cube 2 centre: cell (48, 32, 32) = (0.75, 0.50, 0.50) m
// ─────────────────────────────────────────────────────────────────────────────

namespace mpm
{
namespace test
{

class MPMTestScene
{
public:
    constexpr static std::string_view kTestName_ = "colliding_cubes";

    // ── material ──────────────────────────────────────────────────────────────
    constexpr static float kDx_ = 1.0f / 64.f;
    constexpr static auto  kConstitutiveModel_ = MPMConstitutiveModel::kFixedCorotated;
    constexpr static float kE_      = 1e4f;
    constexpr static float kNu_     = 0.4f;
    constexpr static float kLambda_ = ComputeLameParameters<float>(kE_, kNu_)[0];
    constexpr static float kMu_     = ComputeLameParameters<float>(kE_, kNu_)[1];
    constexpr static auto  kMaterial_ =
        MPMMaterial<kConstitutiveModel_>{
            MPMMaterial<kConstitutiveModel_>::FixedCorotatedMaterialParameter{kLambda_, kMu_}
        };

    constexpr static float kRho_            = 1000.f;
    constexpr static float kParticlePerCell_ = 8.0f;
    constexpr static float kParticleVolume_  = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
    constexpr static float kParticleMass_    = kParticleVolume_ * kRho_;

    // ── time ──────────────────────────────────────────────────────────────────
    constexpr static uint32_t kFps_              = 48;
    constexpr static float    kCfl_              = 0.5f;
    constexpr static float    kDtFactor_         = 1.0f;
    constexpr static float    kTotalSimulatedTime_ = 5.0f;  

    // ── grid / domain ─────────────────────────────────────────────────────────
    // dx=1/64, block_size=4 → cells per dim = blocks×4
    // Need 64 cells for a 1m domain: 16 blocks × 4 cells = 64 cells × (1/64m) = 1.0m ✓
    // (DomainRange<64,64,64> would give 256 cells = 4m domain → particles escape after bounce)
    typedef MPMDomainRange<16, 16, 16>  DomainRange_;
    typedef MPMDomainOffset<0, 0, 0>    DomainOffset_;
    typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
    typedef MPMGridConfig<Domain_>      GridConfig_;

    // ── CRTP config ───────────────────────────────────────────────────────────
    class MPMTestCollidingCubesConfig : public MPMConfigBase<MPMTestCollidingCubesConfig>
    {
    public:
        friend class MPMConfigBase<MPMTestCollidingCubesConfig>;
        constexpr MPMTestCollidingCubesConfig() = default;

    protected:
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDxImpl()                   const -> float    { return kDx_; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockVolumeImpl()          const -> uint32_t { return 64; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockSizeImpl()            const -> uint32_t { return 4; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerCellImpl()   const -> uint32_t { return 128; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBlockImpl()  const -> uint32_t
        {
            return GetBlockVolumeImpl() * GetMaxParticleCountPerCellImpl();
        }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBucketImpl() const -> uint32_t { return 32; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl()       const -> uint32_t { return 1000; }
        // 16³ = 4096 blocks total; cubes occupy ~20 each → 1000 is safe headroom

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl()  const -> uint32_t { return kFps_; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl()   const -> float
        {
            return EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kE_, kNu_, kRho_, kCfl_);
        }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetCflImpl()  const -> float    { return kCfl_; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetTotalSimulatedFrameCountImpl() const -> uint32_t
        {
            return static_cast<uint32_t>(std::round(kTotalSimulatedTime_ * GetFpsImpl()));
        }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetGravityImpl()              const -> float { return 0.f; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMassClampImpl()            const -> float { return 0.f; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistRigidParticleImpl()   const -> bool  { return false; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleCountImpl()   const -> uint32_t { return 0; }

        template<typename Scalar>
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleVelocityImpl() const -> Vector<Scalar, 3>
        { return Vector<Scalar, 3>{0.f, 0.f, 0.f}; }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistIrregularBoundaryImpl() const -> bool { return false; }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ProcessGridCellVelocityImpl(
            const Vector<int, 3>&, Vector<float, 3>&) const -> void { return; }

        MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto UpdateConfigImpl(float, int) -> void {}
    };

    typedef MPMTestCollidingCubesConfig TestConfig_;
};


// ─────────────────────────────────────────────────────────────────────────────
// Scene setup:
//   model 0 = Cube 1  centred at (16, 32, 32)×dx = (0.25, 0.50, 0.50) m
//             velocity = (+0.1, 0, 0) m/s
//   model 1 = Cube 2  centred at (48, 32, 32)×dx = (0.75, 0.50, 0.50) m
//             velocity = (−0.1, 0, 0) m/s
// ─────────────────────────────────────────────────────────────────────────────
auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
{
    std::vector<Vector<float, 3>> position[2];
    std::vector<Vector<float, 3>> velocity[2];

    constexpr float kDx = MPMTestScene::kDx_;

    // Cube centres in physical space
    auto center0 = Vector<float, 3>{16.f, 32.f, 32.f} * kDx;  // (0.25, 0.50, 0.50)
    auto center1 = Vector<float, 3>{48.f, 32.f, 32.f} * kDx;  // (0.75, 0.50, 0.50)

    // Velocities: cubes approach each other along x
    constexpr float kV0 = 0.1f;
    Vector<float, 3> vel0{ kV0, 0.f, 0.f};
    Vector<float, 3> vel1{-kV0, 0.f, 0.f};

    // Loop i,j,k ∈ [−4,+4], 2×2×2 sub-cell → 9³×8 = 5832 particles per cube
    for (int i = -4; i <= 4; ++i)
    for (int j = -4; j <= 4; ++j)
    for (int k = -4; k <= 4; ++k)
    for (int w = 0; w < 8; ++w)
    {
        int di = (w & 4) >> 2;
        int dj = (w & 2) >> 1;
        int dk =  w & 1;

        Vector<float, 3> offset{ i + 0.25f + di * 0.5f,
                                  j + 0.25f + dj * 0.5f,
                                  k + 0.25f + dk * 0.5f };

        position[0].emplace_back(center0 + offset * kDx);
        velocity[0].emplace_back(vel0);

        position[1].emplace_back(center1 + offset * kDx);
        velocity[1].emplace_back(vel1);
    }

    printf("Cube 1 particle count : %zu\n", position[0].size());
    printf("Cube 2 particle count : %zu\n", position[1].size());
    printf("v0 = (+%.3f, 0, 0) / (−%.3f, 0, 0) m/s\n", kV0, kV0);

    particleCount = static_cast<uint32_t>(position[0].size());

    std::vector<MPMModelVariant> modelList;
    using Mat = std::decay_t<decltype(MPMTestScene::kMaterial_)>;
    modelList.push_back({ MPMModel<Mat>{ MPMTestScene::kParticleMass_,
                                         MPMTestScene::kParticleVolume_,
                                         position[0], velocity[0],
                                         MPMTestScene::kMaterial_ } });
    modelList.push_back({ MPMModel<Mat>{ MPMTestScene::kParticleMass_,
                                         MPMTestScene::kParticleVolume_,
                                         position[1], velocity[1],
                                         MPMTestScene::kMaterial_ } });
    return modelList;
}

}  // namespace test
}  // namespace mpm
