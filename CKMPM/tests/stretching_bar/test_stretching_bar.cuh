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
    constexpr static std::string_view kTestName_ = "stretching_bar";

    // ── 网格间距 ──────────────────────────────────────────────────────────
    // dx=1/64, 与 colliding_cubes / dual_rotation / Python 实现保持一致
    constexpr static float kDx_ = 1.0f / 64.f;

    // ── 材料：固定协旋转弹性（与 twisting_bar 完全相同）─────────────────
    constexpr static auto kConstitutiveModel_ = MPMConstitutiveModel::kFixedCorotated;
    constexpr static float kE_      = 1e2f;       // 100 Pa，非常软
    constexpr static float kNu_     = 0.4f;
    constexpr static float kLambda_ = ComputeLameParameters<float>(kE_, kNu_)[0];
    constexpr static float kMu_     = ComputeLameParameters<float>(kE_, kNu_)[1];

    constexpr static auto kMaterial_ =
        MPMMaterial<kConstitutiveModel_>{
            MPMMaterial<kConstitutiveModel_>::FixedCorotatedMaterialParameter{kLambda_, kMu_}
        };

    // ── 粒子参数（ppc=8, 2×2×2 sub-cell）────────────────────────────────
    constexpr static float kRho_             = 2.f;
    constexpr static float kParticlePerCell_ = 8.0f;
    constexpr static float kParticleVolume_  = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
    constexpr static float kParticleMass_    = kParticleVolume_ * kRho_;

    // ── 时间参数 ──────────────────────────────────────────────────────────
    constexpr static uint32_t kFps_               = 48;
    constexpr static float    kCfl_               = 0.5f;
    constexpr static float    kDtFactor_          = 1.0f;
    constexpr static float    kTotalSimulatedTime_ = 5.0f;

    // ── 拉伸速度（两端沿 z 轴相向运动）──────────────────────────────────
    // v_stretch = 0.02 m/s → 5s 后每端各移动 0.1m → bar 伸长 0.2m（从 0.313m 到 0.513m）
    constexpr static float kVStretch_ = 0.02f;

    // ── 网格域：16×16×16 block × 4 cells = 64³ cells = 1m³ ──────────────
    typedef MPMDomainRange<16, 16, 16>             DomainRange_;
    typedef MPMDomainOffset<0, 0, 0>               DomainOffset_;
    typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
    typedef MPMGridConfig<Domain_>                 GridConfig_;

    // ── Bar 几何（与 Python 实现完全一致）────────────────────────────────
    // x: cells [30,35)  →  5 cells = 5/64 ≈ 0.078 m 宽
    // y: cells [30,35)  →  5 cells = 0.078 m 深
    // z: cells [22,42)  →  20 cells = 20/64 ≈ 0.313 m 长
    // Top Dirichlet (+z 拉伸): z ∈ [37,42)
    // Bot Dirichlet (−z 拉伸): z ∈ [22,27)

    class MPMTestStretchingBarConfig : public MPMConfigBase<MPMTestStretchingBarConfig>
    {
    public:
        friend class MPMConfigBase<MPMTestStretchingBarConfig>;
        constexpr MPMTestStretchingBarConfig() = default;

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
        // 5×5×20 格 bar: ceil(6/4)³ × 2 ≈ 54 blocks，留 3× 余量
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl() const -> uint32_t { return 200; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl() const -> uint32_t { return kFps_; }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl() const -> float
        {
            return EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kE_, kNu_, kRho_, kCfl_);
        }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetCflImpl() const -> float { return kCfl_; }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetTotalSimulatedFrameCountImpl() const -> uint32_t
        {
            return static_cast<uint32_t>(std::round(kTotalSimulatedTime_ * GetFpsImpl()));
        }

        // 无重力：纯拉伸，排除重力导致的弯曲
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetGravityImpl() const -> float { return 0.f; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMassClampImpl() const -> float { return 0.f; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistRigidParticleImpl() const -> bool { return false; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleCountImpl() const -> uint32_t { return 0; }

        template<typename Scalar>
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleVelocityImpl() const -> Vector<Scalar, 3>
        {
            return Vector<Scalar, 3>{0.f, 0.f, 0.f};
        }

        // 有 Dirichlet BC：两端 z-translation
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistIrregularBoundaryImpl() const -> bool { return true; }

        // ProcessGridCellVelocityImpl: 对两端区域施加 ±z 拉伸速度
        MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ProcessGridCellVelocityImpl(
            const Vector<int, 3>& cell, Vector<float, 3>& velocity, int frame) const -> void
        {
            // Top end zone: z ∈ [37, 42), x ∈ [29,35], y ∈ [29,35]
            bool inTop = (37 <= cell[2]) && (cell[2] < 42)
                      && (29 <= cell[0]) && (cell[0] <= 35)
                      && (29 <= cell[1]) && (cell[1] <= 35);

            // Bottom end zone: z ∈ [22, 27), x ∈ [29,35], y ∈ [29,35]
            bool inBot = (22 <= cell[2]) && (cell[2] < 27)
                      && (29 <= cell[0]) && (cell[0] <= 35)
                      && (29 <= cell[1]) && (cell[1] <= 35);

            if (inTop)
            {
                velocity = Vector<float, 3>{0.f, 0.f, +kVStretch_};
            }
            else if (inBot)
            {
                velocity = Vector<float, 3>{0.f, 0.f, -kVStretch_};
            }

            // 域边界：清零（Dirichlet wall）
            bool isOutOfBound = (cell[0] < 3) || (cell[1] < 3) || (cell[2] < 3)
                             || (cell[0] >= 61) || (cell[1] >= 61) || (cell[2] >= 61);
            if (isOutOfBound)
            {
                velocity = Vector<float, 3>{0.f, 0.f, 0.f};
            }
        }

        MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto UpdateConfigImpl(float dt, int frame) -> void {}
    };

    typedef MPMTestStretchingBarConfig TestConfig_;
};


auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
{
    constexpr float kDx = MPMTestScene::kDx_;

    std::vector<Vector<float, 3>> position;
    std::vector<Vector<float, 3>> velocity;

    // ── 矩形 bar 粒子生成（2×2×2 sub-cell 采样，ppc=8）──────────────────
    // x: [30,35), y: [30,35), z: [22,42)  →  5×5×20 = 500 格 × 8 = 4000 粒子
    for (int i = 30; i < 35; ++i)
    {
        for (int j = 30; j < 35; ++j)
        {
            for (int k = 22; k < 42; ++k)
            {
                for (int di = 0; di < 2; ++di)
                {
                    for (int dj = 0; dj < 2; ++dj)
                    {
                        for (int dk = 0; dk < 2; ++dk)
                        {
                            position.emplace_back(Vector<float, 3>{
                                (i + 0.25f + di * 0.5f) * kDx,
                                (j + 0.25f + dj * 0.5f) * kDx,
                                (k + 0.25f + dk * 0.5f) * kDx
                            });
                            velocity.emplace_back(Vector<float, 3>{0.f, 0.f, 0.f});
                        }
                    }
                }
            }
        }
    }

    particleCount = static_cast<uint32_t>(position.size());
    printf("[stretching_bar]  particleCount = %u  (5×5×20×8 = 4000)\n", particleCount);

    return { MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{
        MPMTestScene::kParticleMass_,
        MPMTestScene::kParticleVolume_,
        position, velocity,
        MPMTestScene::kMaterial_
    }};
}

} // namespace test
} // namespace mpm
