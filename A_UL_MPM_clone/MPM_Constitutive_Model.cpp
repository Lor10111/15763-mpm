
//!#####################################################################
//! \file MPM_Constitutive_Model.cpp
//!#####################################################################
#include "MPM_Constitutive_Model.h"
using namespace Nova;
//######################################################################
// Constructor
//######################################################################
// template<class T,int d> MPM_Constitutive_Model<T,d>::
// MPM_Constitutive_Model()

//######################################################################
// Destructor
//######################################################################
// template<class T,int d> MPM_Constitutive_Model<T,d>::
// ~MPM_Constitutive_Model()
// {
// }
//######################################################################
// Compute_Lame_Parameters
//######################################################################
template<class T,int d> void MPM_Constitutive_Model<T,d>::
Compute_Lame_Parameters(const T E,const T nu)
{
    mu0=E/((T)2.*((T)1.+nu));
    lambda0=(E*nu)/(((T)1.+nu)*((T)1.-(T)2.*nu));
    if(model==HENCKY) failure_threshold=(T)0.;
}
//######################################################################
// Precompute
//######################################################################
template<class T,int d> void MPM_Constitutive_Model<T,d>::
Precompute(const bool& to_update)
{
    Ue=Matrix<T,d>();Ve=Matrix<T,d>();
    Fe.Fast_Singular_Value_Decomposition(Ue,Se,Ve);
    if(plastic){
        for(int i=0;i<d;++i) if(Se(i)>stretching_yield) Se(i)=stretching_yield;
        for(int i=0;i<d;++i) if(Se(i)<compression_yield) Se(i)=compression_yield;
        Fp=Ve*Se.Inverse()*Ue.Transposed()*Fe*Fp;
        Fe=Ue*Se*Ve.Transposed();
        T power=std::min(hardening_factor*(1-Fp.Determinant()),(T)hardening_max);
        lambda=exp(power)*lambda0;
        mu=exp(power)*mu0;
    }
    else{
        lambda=lambda0;
        mu=mu0;}
    Je=Se.Determinant();
    
    if(model==FIXED_COROTATED){
        Re=Ue*Ve.Transposed();
        He=Fe.Cofactor_Matrix();
    }
    else if(model==HENCKY){
        if(Se(d-1)<=(T)0.){ hencky_psi=FLT_MAX; hencky_P=Mat();}
        else{ Diagonal_Matrix<T,d> log_Se=Se.Logarithm();
        T sum_sqr=sqr(log_Se.Trace());
        hencky_psi=mu*log_Se.Frobenius_Norm_Squared()+(T)0.5*lambda*sum_sqr;
        Diagonal_Matrix<T,d> F_Inv=Se.Inverse();
        Diagonal_Matrix<T,d> P_hat=((T)2*mu*log_Se+lambda*log_Se.Trace())*F_Inv;
        hencky_P=(Ue*P_hat).Times_Transpose(Ve);}
        Evaluate_Hencky_Isotropic_Stress_Derivative(Se,hencky_dP_dF_diagonal);
    }
    else if(model==VONMISES){
        // Log::cout<<"USING VONMISES MODEL"<<std::endl;
        TV epsilon;for(int v=0;v<d;++v) epsilon(v)=std::log(std::max(std::fabs(Se(v,v)),(T)1e-4));
        T trace_epsilon=epsilon.Sum();
        TV epsilon_hat=epsilon-TV(1)*trace_epsilon/d;
        T delta_gamma=epsilon_hat.Norm()-yield_stress/(2.*mu);
        if(delta_gamma>0)
        {
            TV H=epsilon-epsilon_hat*(delta_gamma/epsilon_hat.Norm());
            for(int v=0;v<d;++v)
                Se(v)=std::exp(H(v));
            Fe=Ue*Se*Ve.Transposed();
        }
        Diagonal_Matrix<T,d> log_Se;
        log_Se(0)=std::log(Se(0));
        log_Se(1)=std::log(Se(1));
        if(d==3) log_Se(2)=std::log(Se(2));
        Diagonal_Matrix<T,d> inv_Se=Se.Inverse();
        Diagonal_Matrix<T,d> tmp=inv_Se*log_Se*2.*mu+inv_Se*lambda*log_Se.Trace();
        Matrix<T,d> P; for(int v=0;v<d;++v) P(v,v)=tmp(v);
        vonmises_P=Ue*P*Ve.Transposed();
    }
}
//######################################################################
// Psi
//######################################################################
template<class T,int d> T MPM_Constitutive_Model<T,d>::
Psi() const
{   
    Log::cout<<"RETURNING Psi"<<std::endl;
    if(model==FIXED_COROTATED) return mu*(Se.To_Vector()-1).Norm_Squared()+lambda/2*sqr(Je-1);
    else if(model==HENCKY) return hencky_psi;
    else if(model==VONMISES) return vonmises_psi;
    else {
        Log::cout<<"WRONG"<<std::endl;
        return 0.;
    }
}
//######################################################################
// P
//######################################################################
template<class T,int d> Matrix<T,d> MPM_Constitutive_Model<T,d>::
P()
{
    if(model==FIXED_COROTATED)
        return (T)2.*mu*(Fe-Re)+lambda*(Je-1)*He;
    else if(model==HENCKY) 
        return hencky_P;
    else if(model==VONMISES)
        return vonmises_P;
    else {
        Log::cout<<"WRONG"<<std::endl;
        return Matrix<T,d>();
    }
}
//######################################################################
// Times_dP_dF
//######################################################################
template<class T,int d> Matrix<T,d> MPM_Constitutive_Model<T,d>::
Times_dP_dF(const Matrix<T,d>& dF) const
{
    if(model==FIXED_COROTATED){
        Matrix<T,d> dHdF_times_dF=Times_Cofactor_Matrix_Derivative(Fe,dF);
        return (T)2.*mu*(dF-dR(dF,Re,Fe))+lambda*(He.Times_Transpose(dF)).Trace()*He+lambda*(Je-(T)1.)*dHdF_times_dF;}
    else if(model==HENCKY){
        Matrix<T,d> UtdFV=Ue.Transposed()*dF*Ve;
        Matrix<T,d> action=hencky_dP_dF_diagonal.Differential(UtdFV);
        return (Ue*action).Times_Transpose(Ve);}
    else if(model==VONMISES){
        Log::cout<<"##################################"<<std::endl;
        Log::cout<<"SHOULD NOT BE ACCESSED: Times_dP_dF"<<std::endl;
        Log::cout<<"##################################"<<std::endl;
        return Matrix<T,d>();
    }
    else return Matrix<T,d>();
}
//######################################################################
// Test
//######################################################################
template<class T,int d> void MPM_Constitutive_Model<T,d>::
Test()
{

}
//######################################################################
template class Nova::MPM_Constitutive_Model<float,2>;
template class Nova::MPM_Constitutive_Model<float,3>;
#ifdef COMPILE_WITH_DOUBLE_SUPPORT
template class Nova::MPM_Constitutive_Model<double,2>;
template class Nova::MPM_Constitutive_Model<double,3>;
#endif


