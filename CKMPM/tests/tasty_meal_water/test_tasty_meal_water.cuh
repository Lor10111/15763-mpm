#pragma once

#include "mpm_test_base.h"
#include "mpm_material.cuh"
#include "mpm_model.h"
#include "mpm_domain.h"
#include "mpm_engine.cuh"
#include "data_type.cuh"
#include "mpm_config.h"

#include <array>
#include <cmath>
#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

// ─────────────────────────────────────────────────────────────────────────────
// Tasty Meal — Water variant (Experiment 5: Complex 3D multi-body collision)
//
//   "Drop an ice cube into a cup of water." Three heterogeneous materials in
//   contact: rigid-like elastic cup (FixedCorotated), weakly compressible
//   fluid water (Tait), and rigid-like elastic ice (FixedCorotated).
//
//   Targets: contact precision vs. kernel compactness. Compact kernels (CKMPM)
//   should preserve the sharp ice/water interface and produce more distinct
//   splash droplets than the wider quadratic-B-spline support of MLS-MPM and
//   basic MPM.
//
//   Geometry (1 m³ domain, dx = 1/128):
//     • Cup wall   : annulus  r ∈ [0.060, 0.070],  y ∈ [0.062, 0.170]
//     • Cup floor  : disk     r ≤  0.070,          y ∈ [0.050, 0.062]
//     • Water      : cylinder r ≤  0.058,          y ∈ [0.064, 0.100]
//     • Ice cube   : 0.04 m cube, centre (0.5, 0.30, 0.5), v0 = (0, −5, 0) m/s
//     (cup centred at (cx, _, cz) = (0.5, _, 0.5))
//
//   Boundary handling:
//     • Cup-floor anchor (y < 0.062 ∧ r ≤ 0.075) → velocity = 0
//     • Domain wall (4-cell margin) → reflective (one-sided)
//
//   Time:     fps = 48,  total_time = 1.5 s  →  72 frames
//   Gravity:  −9.8 m/s²
// ─────────────────────────────────────────────────────────────────────────────

namespace mpm
{
namespace test
{

class MPMTestScene
{
public:
    constexpr static std::string_view kTestName_ = "tasty_meal_water";

    // ── Grid spacing ─────────────────────────────────────────────────────────
    constexpr static float kDx_ = 1.0f / 128.f;

    // ── Cup material: stiff Fixed Corotated ──────────────────────────────────
    constexpr static auto  kCupModel_  = MPMConstitutiveModel::kFixedCorotated;
    constexpr static float kCupE_      = 1e7f;
    constexpr static float kCupNu_     = 0.4f;
    constexpr static float kCupLambda_ = ComputeLameParameters<float>(kCupE_, kCupNu_)[0];
    constexpr static float kCupMu_     = ComputeLameParameters<float>(kCupE_, kCupNu_)[1];
    constexpr static auto  kCupMaterial_ = MPMMaterial<kCupModel_>{
        MPMMaterial<kCupModel_>::FixedCorotatedMaterialParameter{kCupLambda_, kCupMu_}};
    constexpr static float kCupRho_    = 2000.f;

    // ── Water material: weakly compressible fluid (Tait EOS) ────────────────
    constexpr static auto  kWaterModel_  = MPMConstitutiveModel::kFluid;
    constexpr static float kWaterBulk_   = 10.f;
    constexpr static float kWaterGamma_  = 7.15f;
    constexpr static float kWaterVisco_  = 0.1f;
    constexpr static auto  kWaterMaterial_ = MPMMaterial<kWaterModel_>{
        MPMMaterial<kWaterModel_>::FluidMaterialParameter{kWaterBulk_, kWaterGamma_, kWaterVisco_}};
    constexpr static float kWaterRho_    = 1000.f;

    // ── Ice material: stiff Fixed Corotated ──────────────────────────────────
    constexpr static auto  kIceModel_  = MPMConstitutiveModel::kFixedCorotated;
    constexpr static float kIceE_      = 1e7f;
    constexpr static float kIceNu_     = 0.4f;
    constexpr static float kIceLambda_ = ComputeLameParameters<float>(kIceE_, kIceNu_)[0];
    constexpr static float kIceMu_     = ComputeLameParameters<float>(kIceE_, kIceNu_)[1];
    constexpr static auto  kIceMaterial_ = MPMMaterial<kIceModel_>{
        MPMMaterial<kIceModel_>::FixedCorotatedMaterialParameter{kIceLambda_, kIceMu_}};
    constexpr static float kIceRho_    = 900.f;

    // ── Particle sampling (8 per cell) ───────────────────────────────────────
    constexpr static float kParticlePerCell_   = 8.0f;
    constexpr static float kParticleVolume_    = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
    constexpr static float kCupParticleMass_   = kParticleVolume_ * kCupRho_;
    constexpr static float kWaterParticleMass_ = kParticleVolume_ * kWaterRho_;
    constexpr static float kIceParticleMass_   = kParticleVolume_ * kIceRho_;

    // ── Time ─────────────────────────────────────────────────────────────────
    constexpr static uint32_t kFps_              = 48;
    constexpr static float    kCfl_              = 0.5f;
    constexpr static float    kDtFactor_         = 1.0f;
    constexpr static float    kTotalSimulatedTime_ = 1.5f;   // 72 frames

    // ── Geometry constants (m) — kept here for cross-impl parity ────────────
    constexpr static float kCupCx_       = 0.5f;
    constexpr static float kCupCz_       = 0.5f;
    constexpr static float kCupRInner_   = 0.060f;
    constexpr static float kCupROuter_   = 0.070f;
    constexpr static float kCupYBottom_  = 0.050f;
    constexpr static float kCupYTop_     = 0.170f;
    constexpr static float kCupFloorThk_ = 0.012f;
    constexpr static float kWaterFillY_  = 0.100f;
    constexpr static float kIceCubeCx_   = 0.5f;
    constexpr static float kIceCubeCy_   = 0.300f;
    constexpr static float kIceCubeCz_   = 0.5f;
    constexpr static float kIceCubeHalf_ = 0.020f;
    constexpr static float kIceVx_       = 0.f;
    constexpr static float kIceVy_       = -5.f;
    constexpr static float kIceVz_       = 0.f;

    // ── Grid / domain: 32 × 32 × 32 blocks × 4 cells = 128³ cells = 1 m³ ────
    typedef MPMDomainRange<32, 32, 32>             DomainRange_;
    typedef MPMDomainOffset<0, 0, 0>               DomainOffset_;
    typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
    typedef MPMGridConfig<Domain_>                 GridConfig_;

    class MPMTastyMealWaterConfig : public MPMConfigBase<MPMTastyMealWaterConfig>
    {
    public:
        friend class MPMConfigBase<MPMTastyMealWaterConfig>;
        constexpr MPMTastyMealWaterConfig() = default;

    protected:
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDxImpl() const -> float { return kDx_; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockVolumeImpl() const -> uint32_t { return 64; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockSizeImpl()   const -> uint32_t { return 4; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerCellImpl() const -> uint32_t { return 32; }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBlockImpl() const -> uint32_t
        {
            return GetBlockVolumeImpl() * GetMaxParticleCountPerCellImpl();
        }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBucketImpl() const -> uint32_t { return 32; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl()       const -> uint32_t { return 60000; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl() const -> uint32_t { return kFps_; }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl() const -> float
        {
            // dt limited by stiffest material (ice / cup share the same E)
            auto dt_ice   = EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kIceE_, kIceNu_, kIceRho_,   kCfl_);
            auto dt_cup   = EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kCupE_, kCupNu_, kCupRho_,   kCfl_);
            return dt_ice < dt_cup ? dt_ice : dt_cup;
        }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetCflImpl() const -> float { return kCfl_; }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetTotalSimulatedFrameCountImpl() const -> uint32_t
        {
            return static_cast<uint32_t>(std::round(kTotalSimulatedTime_ * GetFpsImpl()));
        }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetGravityImpl()    const -> float { return -9.8f; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMassClampImpl()  const -> float { return kWaterParticleMass_ * 1e-8f; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistRigidParticleImpl() const -> bool { return false; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleCountImpl() const -> uint32_t { return 0; }

        template<typename Scalar>
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleVelocityImpl() const -> Vector<Scalar, 3>
        { return Vector<Scalar, 3>{0.f, 0.f, 0.f}; }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistIrregularBoundaryImpl() const -> bool { return true; }

        // Cup floor anchor + reflective domain walls
        MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ProcessGridCellVelocityImpl(
            const Vector<int, 3>& cell, Vector<float, 3>& velocity, int frame) const -> void
        {
            constexpr int kRangeDim[] = {DomainRange_::kDim_[0], DomainRange_::kDim_[1], DomainRange_::kDim_[2]};

            float px = static_cast<float>(cell[0]) * kDx_;
            float py = static_cast<float>(cell[1]) * kDx_;
            float pz = static_cast<float>(cell[2]) * kDx_;

            // 1. Cup-floor anchor (a thin disk at the cup base — keeps cup in place)
            float dxr = px - kCupCx_;
            float dzr = pz - kCupCz_;
            float r2  = dxr * dxr + dzr * dzr;
            float anchorR2 = (kCupROuter_ + 0.005f) * (kCupROuter_ + 0.005f);
            bool inAnchor = (py >= kCupYBottom_ - 0.001f) &&
                            (py <  kCupYBottom_ + kCupFloorThk_) &&
                            (r2  <= anchorR2);
            if (inAnchor)
            {
                velocity = Vector<float, 3>{0.f, 0.f, 0.f};
                return;
            }

            // 2. Reflective domain walls (4-cell margin)
            Vector<float, 3> normal{0.f, 0.f, 0.f};
            bool onBoundary = false;
            for (int i = 0; i < 3; ++i)
            {
                if (cell[i] < 4)                          { normal[i] =  1.f; onBoundary = true; }
                else if (cell[i] >= kRangeDim[i] * 4 - 4) { normal[i] = -1.f; onBoundary = true; }
            }
            if (onBoundary)
            {
                float n = normal.Norm();
                if (n > 0.f)
                {
                    normal = normal / n;
                    float magnitude = normal.Dot(velocity);
                    if (magnitude < 0.f) velocity -= magnitude * normal;
                }
            }
        }

        MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto UpdateConfigImpl(float dt, int frame) -> void {}
    };

    typedef MPMTastyMealWaterConfig TestConfig_;
};


// ─────────────────────────────────────────────────────────────────────────────
// SetupModel — programmatic generation of cup, water, ice particle clouds.
//
//   Output order:
//     model 0 : cup   (FixedCorotated)
//     model 1 : water (Fluid / Tait EOS)
//     model 2 : ice   (FixedCorotated)
// ─────────────────────────────────────────────────────────────────────────────
auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
{
    constexpr float dx       = MPMTestScene::kDx_;
    constexpr float cupCx    = MPMTestScene::kCupCx_;
    constexpr float cupCz    = MPMTestScene::kCupCz_;
    constexpr float rInner   = MPMTestScene::kCupRInner_;
    constexpr float rOuter   = MPMTestScene::kCupROuter_;
    constexpr float yBot     = MPMTestScene::kCupYBottom_;
    constexpr float yTop     = MPMTestScene::kCupYTop_;
    constexpr float floorThk = MPMTestScene::kCupFloorThk_;
    constexpr float waterTop = MPMTestScene::kWaterFillY_;
    constexpr float iceCx    = MPMTestScene::kIceCubeCx_;
    constexpr float iceCy    = MPMTestScene::kIceCubeCy_;
    constexpr float iceCz    = MPMTestScene::kIceCubeCz_;
    constexpr float iceHalf  = MPMTestScene::kIceCubeHalf_;

    std::vector<MPMModelVariant> modelList;

    // ─── Cup particles (annular wall + disk floor) ───────────────────────────
    {
        std::vector<Vector<float, 3>> position;
        std::vector<Vector<float, 3>> velocity;
        velocity.emplace_back(Vector<float, 3>{0.f, 0.f, 0.f});

        const int iMin = static_cast<int>((cupCx - rOuter) / dx) - 1;
        const int iMax = static_cast<int>((cupCx + rOuter) / dx) + 2;
        const int jMin = static_cast<int>(yBot / dx) - 1;
        const int jMax = static_cast<int>(yTop / dx) + 2;
        const int kMin = static_cast<int>((cupCz - rOuter) / dx) - 1;
        const int kMax = static_cast<int>((cupCz + rOuter) / dx) + 2;

        for (int i = iMin; i < iMax; ++i)
        for (int j = jMin; j < jMax; ++j)
        for (int k = kMin; k < kMax; ++k)
        for (int w = 0; w < 8; ++w)
        {
            int di = (w & 4) >> 2;
            int dj = (w & 2) >> 1;
            int dk =  w & 1;
            float px = (i + 0.25f + di * 0.5f) * dx;
            float py = (j + 0.25f + dj * 0.5f) * dx;
            float pz = (k + 0.25f + dk * 0.5f) * dx;

            float dxr = px - cupCx;
            float dzr = pz - cupCz;
            float r   = std::sqrt(dxr * dxr + dzr * dzr);

            bool inWall  = (r >= rInner) && (r <= rOuter) &&
                           (py >= yBot + floorThk) && (py <= yTop);
            bool inFloor = (r <= rOuter) &&
                           (py >= yBot) && (py <  yBot + floorThk);
            if (inWall || inFloor)
            {
                position.emplace_back(Vector<float, 3>{px, py, pz});
            }
        }

        printf("[tasty_meal_water] Cup   particles : %zu\n", position.size());
        particleCount += static_cast<uint32_t>(position.size());

        modelList.emplace_back(MPMModel<std::decay_t<decltype(MPMTestScene::kCupMaterial_)>>{
            MPMTestScene::kCupParticleMass_,
            MPMTestScene::kParticleVolume_,
            position, velocity,
            MPMTestScene::kCupMaterial_});
    }

    // ─── Water particles (cylindrical column inside cup) ─────────────────────
    {
        std::vector<Vector<float, 3>> position;
        std::vector<Vector<float, 3>> velocity;
        velocity.emplace_back(Vector<float, 3>{0.f, 0.f, 0.f});

        const float waterR    = rInner - 0.002f;            // 2 mm clearance from inner wall
        const float waterYBot = yBot + floorThk + 0.002f;   // 2 mm above cup floor

        const int iMin = static_cast<int>((cupCx - waterR) / dx) - 1;
        const int iMax = static_cast<int>((cupCx + waterR) / dx) + 2;
        const int jMin = static_cast<int>(waterYBot / dx) - 1;
        const int jMax = static_cast<int>(waterTop  / dx) + 2;
        const int kMin = static_cast<int>((cupCz - waterR) / dx) - 1;
        const int kMax = static_cast<int>((cupCz + waterR) / dx) + 2;

        for (int i = iMin; i < iMax; ++i)
        for (int j = jMin; j < jMax; ++j)
        for (int k = kMin; k < kMax; ++k)
        for (int w = 0; w < 8; ++w)
        {
            int di = (w & 4) >> 2;
            int dj = (w & 2) >> 1;
            int dk =  w & 1;
            float px = (i + 0.25f + di * 0.5f) * dx;
            float py = (j + 0.25f + dj * 0.5f) * dx;
            float pz = (k + 0.25f + dk * 0.5f) * dx;

            float dxr = px - cupCx;
            float dzr = pz - cupCz;
            float r   = std::sqrt(dxr * dxr + dzr * dzr);
            if (r <= waterR && py >= waterYBot && py <= waterTop)
            {
                position.emplace_back(Vector<float, 3>{px, py, pz});
            }
        }

        printf("[tasty_meal_water] Water particles : %zu\n", position.size());
        particleCount += static_cast<uint32_t>(position.size());

        modelList.emplace_back(MPMModel<std::decay_t<decltype(MPMTestScene::kWaterMaterial_)>>{
            MPMTestScene::kWaterParticleMass_,
            MPMTestScene::kParticleVolume_,
            position, velocity,
            MPMTestScene::kWaterMaterial_});
    }

    // ─── Ice cube particles ──────────────────────────────────────────────────
    {
        std::vector<Vector<float, 3>> position;
        std::vector<Vector<float, 3>> velocity;
        velocity.emplace_back(Vector<float, 3>{MPMTestScene::kIceVx_,
                                                MPMTestScene::kIceVy_,
                                                MPMTestScene::kIceVz_});

        const int iMin = static_cast<int>((iceCx - iceHalf) / dx) - 1;
        const int iMax = static_cast<int>((iceCx + iceHalf) / dx) + 2;
        const int jMin = static_cast<int>((iceCy - iceHalf) / dx) - 1;
        const int jMax = static_cast<int>((iceCy + iceHalf) / dx) + 2;
        const int kMin = static_cast<int>((iceCz - iceHalf) / dx) - 1;
        const int kMax = static_cast<int>((iceCz + iceHalf) / dx) + 2;

        for (int i = iMin; i < iMax; ++i)
        for (int j = jMin; j < jMax; ++j)
        for (int k = kMin; k < kMax; ++k)
        for (int w = 0; w < 8; ++w)
        {
            int di = (w & 4) >> 2;
            int dj = (w & 2) >> 1;
            int dk =  w & 1;
            float px = (i + 0.25f + di * 0.5f) * dx;
            float py = (j + 0.25f + dj * 0.5f) * dx;
            float pz = (k + 0.25f + dk * 0.5f) * dx;

            if (std::abs(px - iceCx) <= iceHalf &&
                std::abs(py - iceCy) <= iceHalf &&
                std::abs(pz - iceCz) <= iceHalf)
            {
                position.emplace_back(Vector<float, 3>{px, py, pz});
            }
        }

        printf("[tasty_meal_water] Ice   particles : %zu\n", position.size());
        particleCount += static_cast<uint32_t>(position.size());

        modelList.emplace_back(MPMModel<std::decay_t<decltype(MPMTestScene::kIceMaterial_)>>{
            MPMTestScene::kIceParticleMass_,
            MPMTestScene::kParticleVolume_,
            position, velocity,
            MPMTestScene::kIceMaterial_});
    }

    return modelList;
}

}  // namespace test
}  // namespace mpm
