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

namespace mpm
{
namespace test
{

class MPMTestScene
{
public:
    constexpr static std::string_view kTestName_ = "dropping_block_sand";

    // ── Grid spacing ──────────────────────────────────────────────────────
    constexpr static float kDx_ = 1.0f / 256.f;

    // ── Material: Drucker-Prager sand ────────────────────────────────────
    constexpr static auto kConstitutiveModel_ = MPMConstitutiveModel::kDruckerPragerStvkhencky;
    constexpr static float kE_      = 1e4f;
    constexpr static float kNu_     = 0.4f;
    constexpr static float kLambda_ = ComputeLameParameters<float>(kE_, kNu_)[0];
    constexpr static float kMu_     = ComputeLameParameters<float>(kE_, kNu_)[1];

    constexpr static auto kMaterial_ =
        MPMMaterial<kConstitutiveModel_>{
            MPMMaterial<kConstitutiveModel_>::DruckerPragerStvkhenckyMaterialParameter{kLambda_, kMu_}
        };

    // ── Particle parameters ───────────────────────────────────────────────
    constexpr static float kRho_             = 2000.f;
    constexpr static float kParticlePerCell_ = 8.0f;
    constexpr static float kParticleVolume_  = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
    constexpr static float kParticleMass_    = kParticleVolume_ * kRho_;

    // ── Time parameters ───────────────────────────────────────────────────
    constexpr static uint32_t kFps_               = 48;
    constexpr static float    kTotalSimulatedTime_ = 5.0f;

    // ── Domain: 64×64×64 blocks = 256³ cells = 1m³ ───────────────────────
    typedef MPMDomainRange<64, 64, 64>             DomainRange_;
    typedef MPMDomainOffset<0, 0, 0>               DomainOffset_;
    typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
    typedef MPMGridConfig<Domain_>                 GridConfig_;

    class MPMTestDroppingBlockSandConfig : public MPMConfigBase<MPMTestDroppingBlockSandConfig>
    {
    public:
        friend class MPMConfigBase<MPMTestDroppingBlockSandConfig>;
        constexpr MPMTestDroppingBlockSandConfig() = default;

    protected:
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDxImpl() const -> float { return kDx_; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockVolumeImpl() const -> uint32_t { return 64; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockSizeImpl() const -> uint32_t { return 4; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerCellImpl() const -> uint32_t { return 128; }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBlockImpl() const -> uint32_t
        {
            return GetBlockVolumeImpl() * GetMaxParticleCountPerCellImpl();
        }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBucketImpl() const -> uint32_t { return 32; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl() const -> uint32_t { return 50000; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl() const -> uint32_t { return kFps_; }

        // Fixed timestep for sand (CFL-computed dt can be too small with high stiffness)
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl() const -> float { return 5e-5f; }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetCflImpl() const -> float { return 0.5f; }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetTotalSimulatedFrameCountImpl() const -> uint32_t
        {
            return static_cast<uint32_t>(std::round(kTotalSimulatedTime_ * GetFpsImpl()));
        }

        // Gravity along -y axis
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetGravityImpl() const -> float { return -2.f; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMassClampImpl() const -> float { return 0.f; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistRigidParticleImpl() const -> bool { return false; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleCountImpl() const -> uint32_t { return 0; }

        template<typename Scalar>
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleVelocityImpl() const -> Vector<Scalar, 3>
        {
            return Vector<Scalar, 3>{0.f, 0.f, 0.f};
        }

        // Irregular boundary: needed for the rigid filter plate
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistIrregularBoundaryImpl() const -> bool { return true; }

        // Filter plate: y ∈ [100, 108], solid except 3 circular holes at x=(64,128,192), z=128, r=16
        // When irregular boundary is on, domain wall handling is NOT automatic — must be done here.
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ProcessGridCellVelocityImpl(
            const Vector<int, 3>& cell, Vector<float, 3>& velocity, int frame) const -> void
        {
            // 1. Domain wall boundary (4-cell margin)
            bool isOutOfBound = (cell[0] < 4) || (cell[1] < 4) || (cell[2] < 4) ||
                                (cell[0] >= 252) || (cell[1] >= 252) || (cell[2] >= 252);
            if (isOutOfBound) { velocity = Vector<float, 3>{0.f, 0.f, 0.f}; return; }

            // 2. Rigid filter plate at y ∈ [100, 108]
            bool inFilterLayer = (cell[1] >= 100 && cell[1] < 108);
            if (inFilterLayer)
            {
                int dx0 = cell[0] - 64,  dz0 = cell[2] - 128;
                int dx1 = cell[0] - 128, dz1 = cell[2] - 128;
                int dx2 = cell[0] - 192, dz2 = cell[2] - 128;
                constexpr int R2 = 16 * 16;
                bool inHole = (dx0*dx0 + dz0*dz0 <= R2) ||
                              (dx1*dx1 + dz1*dz1 <= R2) ||
                              (dx2*dx2 + dz2*dz2 <= R2);
                if (!inHole) { velocity = Vector<float, 3>{0.f, 0.f, 0.f}; }
            }
        }

        MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto UpdateConfigImpl(float dt, int frame) -> void {}
    };

    typedef MPMTestDroppingBlockSandConfig TestConfig_;
};


auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
{
    constexpr float kDx = MPMTestScene::kDx_;

    // Block center: x=128, y=185, z=128 cells (bottom at y=170, filter top at y=108 → gap=62 cells≈0.24m)
    auto center = Vector<float, 3>{128.f, 185.f, 128.f} * kDx;

    std::vector<Vector<float, 3>> position;
    std::vector<Vector<float, 3>> velocity;

    // 30×30×30 cells, 8 particles/cell = 216,000 particles total
    for (int i = -15; i < 15; ++i)
    {
        for (int j = -15; j < 15; ++j)
        {
            for (int k = -15; k < 15; ++k)
            {
                for (int w = 0; w < 8; ++w)
                {
                    // 2×2×2 sub-cell sampling per cell
                    int di = (w & 4) >> 2;
                    int dj = (w & 2) >> 1;
                    int dk =  w & 1;

                    auto pos = center + Vector<float, 3>{
                        i + 0.25f + di * 0.5f,
                        j + 0.25f + dj * 0.5f,
                        k + 0.25f + dk * 0.5f
                    } * kDx;

                    position.emplace_back(pos);
                    velocity.emplace_back(Vector<float, 3>{0.f, 0.f, 0.f});
                }
            }
        }
    }

    printf("Sand particle count: %zu\n", position.size());

    particleCount = static_cast<uint32_t>(position.size());

    std::vector<MPMModelVariant> modelList;
    modelList.push_back({
        MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{
            MPMTestScene::kParticleMass_,
            MPMTestScene::kParticleVolume_,
            position, velocity,
            MPMTestScene::kMaterial_
        }
    });

    return modelList;
}

} // namespace test
} // namespace mpm
