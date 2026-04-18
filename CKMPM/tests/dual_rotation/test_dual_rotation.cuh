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
    constexpr static std::string_view kTestName_ = "dual_rotation";

    // ── 网格间距 ──────────────────────────────────────────────────────────
    // dx=1/64 与 colliding_cubes / MLS-MPM / Basic-MPM-3D 保持一致
    constexpr static float kDx_ = 1.0f / 64.f;

    // ── 材料：固定协旋转弹性 ──────────────────────────────────────────────
    constexpr static auto kConstitutiveModel_ = MPMConstitutiveModel::kFixedCorotated;
    constexpr static float kE_      = 1e6f;
    constexpr static float kNu_     = 0.4f;
    constexpr static float kLambda_ = ComputeLameParameters<float>(kE_, kNu_)[0];
    constexpr static float kMu_     = ComputeLameParameters<float>(kE_, kNu_)[1];

    constexpr static auto kMaterial_ =
        MPMMaterial<kConstitutiveModel_>{
            MPMMaterial<kConstitutiveModel_>::FixedCorotatedMaterialParameter{kLambda_, kMu_}
        };

    // ── 粒子参数 ──────────────────────────────────────────────────────────
    constexpr static float kRho_             = 1000.f;
    constexpr static float kParticlePerCell_ = 8.0f;
    constexpr static float kParticleVolume_  = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
    constexpr static float kParticleMass_    = kParticleVolume_ * kRho_;

    // ── 时间参数 ──────────────────────────────────────────────────────────
    constexpr static uint32_t kFps_               = 48;
    constexpr static float    kCfl_               = 0.5f;
    constexpr static float    kDtFactor_          = 1.0f;
    constexpr static float    kTotalSimulatedTime_ = 5.0f;

    // ── 角速度大小（rad/s）────────────────────────────────────────────────
    // 方块半径约 4*sqrt(2)*kDx ≈ 0.089 m，最大线速度 ≈ kOmega * 0.089 ≈ 0.18 m/s
    constexpr static float kOmega_ = 2.0f;

    // ── 网格域：16×16×16 个 block × 4 cells/block = 64³ 格 × (1/64 m) = 1m³ ──
    // 与 colliding_cubes 完全相同，避免粒子飞出活跃 block 范围
    typedef MPMDomainRange<16, 16, 16>             DomainRange_;
    typedef MPMDomainOffset<0, 0, 0>               DomainOffset_;
    typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
    typedef MPMGridConfig<Domain_>                 GridConfig_;

    class MPMTestDualRotationConfig : public MPMConfigBase<MPMTestDualRotationConfig>
    {
    public:
        friend class MPMConfigBase<MPMTestDualRotationConfig>;
        constexpr MPMTestDualRotationConfig() = default;

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
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl() const -> uint32_t { return 400; }
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

        // 无重力：纯旋转运动，不受外力
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetGravityImpl() const -> float { return 0.f; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMassClampImpl() const -> float { return 0.f; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistRigidParticleImpl() const -> bool { return false; }
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleCountImpl() const -> uint32_t { return 0; }

        template<typename Scalar>
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleVelocityImpl() const -> Vector<Scalar, 3>
        {
            return Vector<Scalar, 3>{0.f, 0.f, 0.f};
        }

        // 无 Dirichlet 边界：两个方块之间不施加任何约束，自由演化
        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistIrregularBoundaryImpl() const -> bool { return false; }

        constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ProcessGridCellVelocityImpl(
            const Vector<int, 3>& cell, Vector<float, 3>& velocity) const -> void
        {
            return;
        }

        MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto UpdateConfigImpl(float dt, int frame) -> void {}
    };

    typedef MPMTestDualRotationConfig TestConfig_;
};


auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
{
    constexpr float kDx    = MPMTestScene::kDx_;
    constexpr float kOmega = MPMTestScene::kOmega_;

    // ── 两个方块的中心 ────────────────────────────────────────────────────
    // dx=1/64，64³ 格，域大小 = 1m³
    auto center0 = Vector<float, 3>{20.f, 32.f, 32.f} * kDx;  // 方块1 中心 (0.3125, 0.5, 0.5)
    auto center1 = Vector<float, 3>{44.f, 32.f, 32.f} * kDx;  // 方块2 中心 (0.6875, 0.5, 0.5)

    std::vector<Vector<float, 3>> position[2];
    std::vector<Vector<float, 3>> velocity[2];

    Vector<float, 3> initAngularMomentum0{0.f, 0.f, 0.f};  // 方块1 初始角动量
    Vector<float, 3> initAngularMomentum1{0.f, 0.f, 0.f};  // 方块2 初始角动量

    // ── 生成两个方块的粒子 ────────────────────────────────────────────────
    for (int i = -4; i <= 4; ++i)
    {
        for (int j = -4; j <= 4; ++j)
        {
            for (int k = -4; k <= 4; ++k)
            {
                for (int w = 0; w < 8; ++w)
                {
                    // 每格内 2×2×2 sub-cell 采样，避免粒子落在格边界
                    int di = (w & 4) >> 2;
                    int dj = (w & 2) >> 1;
                    int dk =  w & 1;

                    auto offset = Vector<float, 3>{
                        i - 0.25f + di * 0.5f,
                        j - 0.25f + dj * 0.5f,
                        k - 0.25f + dk * 0.5f
                    } * kDx;

                    // ── 方块1：绕 z 轴正向旋转，ω = (0, 0, +kOmega) ────────
                    // v = ω × r，ω=(0,0,ω₀), r=(rx,ry,rz)
                    // → v = (-ω₀*ry, +ω₀*rx, 0)
                    {
                        auto pos = center0 + offset;
                        auto r   = pos - center0;  // 相对于方块中心的位移
                        auto vel = Vector<float, 3>{
                            -kOmega * r[1],   // vx = -ω * ry
                            +kOmega * r[0],   // vy = +ω * rx
                             0.f              // vz = 0（绕 z 轴旋转，z 速度为 0）
                        };
                        position[0].emplace_back(pos);
                        velocity[0].emplace_back(vel);
                        // L = m * (r × v)，只追踪 z 分量：L_z = m*(rx*vy - ry*vx)
                        initAngularMomentum0[2] += MPMTestScene::kParticleMass_ *
                            (r[0] * vel[1] - r[1] * vel[0]);
                    }

                    // ── 方块2：绕 z 轴反向旋转，ω = (0, 0, -kOmega) ────────
                    // → v = (+ω₀*ry, -ω₀*rx, 0)
                    {
                        auto pos = center1 + offset;
                        auto r   = pos - center1;
                        auto vel = Vector<float, 3>{
                            +kOmega * r[1],   // vx = +ω * ry
                            -kOmega * r[0],   // vy = -ω * rx
                             0.f
                        };
                        position[1].emplace_back(pos);
                        velocity[1].emplace_back(vel);
                        initAngularMomentum1[2] += MPMTestScene::kParticleMass_ *
                            (r[0] * vel[1] - r[1] * vel[0]);
                    }
                }
            }
        }
    }


    particleCount = static_cast<uint32_t>(position[0].size());

    // ── 打包成 MPMModelVariant 列表（两个模型）────────────────────────────
    // Simulate() 会对每个 model 分别写 conservationMetric_model_0.bin / _1.bin
    std::vector<MPMModelVariant> modelList;
    modelList.push_back({
        MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{
            MPMTestScene::kParticleMass_,
            MPMTestScene::kParticleVolume_,
            position[0], velocity[0],
            MPMTestScene::kMaterial_
        }
    });
    modelList.push_back({
        MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{
            MPMTestScene::kParticleMass_,
            MPMTestScene::kParticleVolume_,
            position[1], velocity[1],
            MPMTestScene::kMaterial_
        }
    });

    return modelList;
}

} // namespace test
} // namespace mpm
