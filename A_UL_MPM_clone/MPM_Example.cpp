//!#####################################################################
//! \file MPM_Example.cpp
//!#####################################################################
#include <nova/Dynamics/Hierarchy/Grid_Hierarchy_Initializer.h>
#include <nova/Dynamics/Utilities/SPGrid_Flags.h>
#include <nova/SPGrid/Tools/SPGrid_Clear.h>
#include <nova/Tools/Grids/Grid_Iterator_Cell.h>
#include <nova/Tools/Utilities/File_Utilities.h>
#include <nova/Tools/Utilities/Utilities.h>
#include <nova/Tools/Krylov_Solvers/Conjugate_Gradient.h>
#include <omp.h>
#include "MPM_Example.h"
#include "Traverse_Helper.h"
#include "Vector_Normalization_Helper.h"
#include "Explicit_Force_Helper.h"
#include "Grid_Based_Collision_Helper.h"
#include "Flag_Helper.h"
#include "MPM_RHS_Helper.h"
#include "Channel_Vector_Norm_Helper.h"
#include "Compare_Helper.h"
#include "Flag_Setup_Helper.h"

#include <nova/SPGrid/Tools/SPGrid_Arithmetic.h>
#include "AV_Helper.h"
#include "Hourglass_Collision_Helper.h"
#include "TLMPM_Grid_Based_Collision_Helper.h"
#include "MPM_Object.h"

#include "Rigid_Cup.h"
#include "Cup_Collision_Helper.h"

#include "Scalar_Normalization_Helper.h"

#include "Fixed_Domain_Helper.h"
#include "Barrier_Collision_Helper.h"
#include "Sphere_Collision_Helper.h"
#include "Box_Collision_Helper.h"
#include <nova/Geometry/Basic_Geometry/Box.h>

#include "./Implicit_Force_Helper/MPM_CG_Vector.h"
#include "./Implicit_Force_Helper/MPM_CG_System.h"

#include "Fixed_Ring_Collision_Helper.h"


using namespace Nova;
using namespace SPGrid;
//######################################################################
// Constructor
//######################################################################
template<class T> MPM_Example<T,2>::
MPM_Example()
    :Base(),hierarchy(nullptr)
{
    level=0;
    random.Set_Seed(0);

    gravity=TV::Axis_Vector(1)*(T)-2.;

    flip=0.95;

    flags_channel                           = &Struct_type::flags;
    mass_channel                            = &Struct_type::ch0;
    
    velocity_channels(0)                    = &Struct_type::ch1;
    velocity_channels(1)                    = &Struct_type::ch2;
    
    velocity_star_channels(0)               = &Struct_type::ch3;
    velocity_star_channels(1)               = &Struct_type::ch4;
    
    f_channels(0)                           = &Struct_type::ch5;
    f_channels(1)                           = &Struct_type::ch6;

    position_channels(0)                    = &Struct_type::ch7;
    position_channels(1)                    = &Struct_type::ch8;

    barrier_velocity_channels(0)            = &Struct_type::ch9;
    barrier_velocity_channels(1)            = &Struct_type::ch10;

    X0_channels(0)                          = &Struct_type::ch11;
    X0_channels(1)                          = &Struct_type::ch12;

    q_channels(0)                           = &Struct_type::ch13;
    q_channels(1)                           = &Struct_type::ch14;

    s_channels(0)                           = &Struct_type::ch16;
    s_channels(1)                           = &Struct_type::ch17;

    r_channels(0)                           = &Struct_type::ch19;
    r_channels(1)                           = &Struct_type::ch20;

    z_channels(0)                           = &Struct_type::ch22;
    z_channels(1)                           = &Struct_type::ch23;

    rhs_channels(0)                         = &Struct_type::ch24;
    rhs_channels(1)                         = &Struct_type::ch25;

    dt=dt_1=dt_2=1e-4;
}
//######################################################################
// Constructor
//######################################################################
template<class T> MPM_Example<T,3>::
MPM_Example()
    :Base(),hierarchy(nullptr)
{
    level=0;
    random.Set_Seed(0);

    gravity=TV::Axis_Vector(1)*(T)-2.;

    flip=0.95;

    flags_channel                           = &Struct_type::flags;
    mass_channel                            = &Struct_type::ch0;
    
    velocity_channels(0)                    = &Struct_type::ch1;
    velocity_channels(1)                    = &Struct_type::ch2;
    velocity_channels(2)                    = &Struct_type::ch3;
    
    velocity_star_channels(0)               = &Struct_type::ch4;
    velocity_star_channels(1)               = &Struct_type::ch5;
    velocity_star_channels(2)               = &Struct_type::ch6;
    
    f_channels(0)                           = &Struct_type::ch7;
    f_channels(1)                           = &Struct_type::ch8;
    f_channels(2)                           = &Struct_type::ch9;

    position_channels(0)                    = &Struct_type::ch10;
    position_channels(1)                    = &Struct_type::ch11;
    position_channels(2)                    = &Struct_type::ch12;

    barrier_velocity_channels(0)            = &Struct_type::ch13;
    barrier_velocity_channels(1)            = &Struct_type::ch14;
    barrier_velocity_channels(2)            = &Struct_type::ch15;

    q_channels(0)                           = &Struct_type::ch16;
    q_channels(1)                           = &Struct_type::ch17;
    q_channels(2)                           = &Struct_type::ch18;

    s_channels(0)                           = &Struct_type::ch19;
    s_channels(1)                           = &Struct_type::ch20;
    s_channels(2)                           = &Struct_type::ch21;

    r_channels(0)                           = &Struct_type::ch22;
    r_channels(1)                           = &Struct_type::ch23;
    r_channels(2)                           = &Struct_type::ch24;

    z_channels(0)                           = &Struct_type::ch25;
    z_channels(1)                           = &Struct_type::ch26;
    z_channels(2)                           = &Struct_type::ch27;

    rhs_channels(0)                         = &Struct_type::ch28;
    rhs_channels(1)                         = &Struct_type::ch29;
    rhs_channels(2)                         = &Struct_type::ch30;
    dt=dt_1=dt_2=1e-4;
}
//######################################################################
// Destructor
//######################################################################
template<class T> MPM_Example<T,2>::
~MPM_Example()
{
    if(hierarchy!=nullptr) delete hierarchy;
}
//######################################################################
// Destructor
//######################################################################
template<class T> MPM_Example<T,3>::
~MPM_Example()
{
    if(hierarchy!=nullptr) delete hierarchy;
}
//######################################################################
// Set_Dt
//######################################################################
template<class T> void MPM_Example<T,2>::
Set_Dt(const T& dt_input)
{
    dt=dt_input;
}
//######################################################################
// Set_Dt
//######################################################################
template<class T> void MPM_Example<T,3>::
Set_Dt(const T& dt_input)
{
    dt=dt_input;
}
//######################################################################
// Compute_Bounding_Box
//######################################################################
template<class T> void MPM_Example<T,2>::
Compute_Bounding_Box(Range<T,2>& bbox,const int& oid)
{
    Array<TV> min_corner_per_thread(threads,domain.max_corner);
    Array<TV> max_corner_per_thread(threads,domain.min_corner);
    
    const T_Object& obj=objects(oid);
#pragma omp parallel for
    for(unsigned i=0;i<obj.particle_ids.size();++i){const int tid=omp_get_thread_num();
        const int pid=obj.particle_ids[i];
        const T_Particle& p=particles(pid);
        if(p.valid){
            TV& current_min_corner=min_corner_per_thread(tid);
            TV& current_max_corner=max_corner_per_thread(tid);
            const TV& p_X=p.X;
            for(int v=0;v<2;++v){
                const T dd=(T)3./counts(v);
                current_min_corner(v)=std::min(current_min_corner(v),p_X(v)-dd);
                current_max_corner(v)=std::max(current_max_corner(v),p_X(v)+dd);}
            const TV& p_X0=p.X0;
            for(int v=0;v<2;++v){
                const T dd=(T)3./counts(v);
                current_min_corner(v)=std::min(current_min_corner(v),p_X0(v)-dd);
                current_max_corner(v)=std::max(current_max_corner(v),p_X0(v)+dd);}
        }
    }

    for(int v=0;v<2;++v){
        bbox.min_corner(v)=min_corner_per_thread(0)(v);
        bbox.max_corner(v)=max_corner_per_thread(0)(v);
    }

    for(int tid=1;tid<threads;++tid) for(int v=0;v<2;++v){
        bbox.min_corner(v)=std::min(bbox.min_corner(v),min_corner_per_thread(tid)(v));
        bbox.max_corner(v)=std::max(bbox.max_corner(v),max_corner_per_thread(tid)(v));}
    
    for(int v=0;v<2;++v){ 
        bbox.min_corner(v)=std::max(domain.min_corner(v),bbox.min_corner(v));
        bbox.max_corner(v)=std::min(domain.max_corner(v),bbox.max_corner(v));}
}
//######################################################################
// Compute_Bounding_Box
//######################################################################
template<class T> void MPM_Example<T,3>::
Compute_Bounding_Box(Range<T,3>& bbox,const int& oid)
{
    Array<TV> min_corner_per_thread(threads,domain.max_corner);
    Array<TV> max_corner_per_thread(threads,domain.min_corner);
    
    const T_Object& obj=objects(oid);
#pragma omp parallel for
    for(unsigned i=0;i<obj.particle_ids.size();++i){
        const int tid=omp_get_thread_num();
        const int pid=obj.particle_ids[i];
        const T_Particle& p=particles(pid);
        if(p.valid){
            TV& current_min_corner=min_corner_per_thread(tid);
            TV& current_max_corner=max_corner_per_thread(tid);
            const TV& p_X0=p.X0;
            for(int v=0;v<3;++v){
                const T dd=(T)3./counts(v);
                current_min_corner(v)=std::min(current_min_corner(v),p_X0(v)-dd);
                current_max_corner(v)=std::max(current_max_corner(v),p_X0(v)+dd);
            }
            const TV& p_X=p.X;
            for(int v=0;v<3;++v){
                const T dd=(T)3./counts(v);
                current_min_corner(v)=std::min(current_min_corner(v),p_X(v)-dd);
                current_max_corner(v)=std::max(current_max_corner(v),p_X(v)+dd);
            }
        }

    }

    for(int v=0;v<3;++v){
        bbox.min_corner(v)=min_corner_per_thread(0)(v);
        bbox.max_corner(v)=max_corner_per_thread(0)(v);
    }

    for(int tid=1;tid<threads;++tid) for(int v=0;v<3;++v){
        bbox.min_corner(v)=std::min(bbox.min_corner(v),min_corner_per_thread(tid)(v));
        bbox.max_corner(v)=std::max(bbox.max_corner(v),max_corner_per_thread(tid)(v));}
    
    for(int v=0;v<3;++v){ 
        bbox.min_corner(v)=std::max(domain.min_corner(v),bbox.min_corner(v));
        bbox.max_corner(v)=std::min(domain.max_corner(v),bbox.max_corner(v));}
}
//######################################################################
// Rasterize_Voxels: set inside cell
//######################################################################
template<class T> void MPM_Example<T,2>::
Rasterize_Voxels()
{
    using Cell_Iterator             = Grid_Iterator_Cell<T,2>;
    using Hierarchy_Initializer     = Grid_Hierarchy_Initializer<Struct_type,T,2>;
    const Grid<T,2>& grid=hierarchy->Lattice(level);
    Range<int,2> bounding_grid_cells(grid.Clamp_To_Cell(bbox.min_corner),grid.Clamp_To_Cell(bbox.max_corner));
    for(Cell_Iterator iterator(grid,bounding_grid_cells);iterator.Valid();iterator.Next())
        hierarchy->Activate_Cell(0,iterator.Cell_Index(),Cell_Type_Dirichlet);
    
    hierarchy->Update_Block_Offsets();
    hierarchy->Initialize_Red_Black_Partition(2*threads);
}
//######################################################################
// Rasterize_Voxels: set inside cell
//######################################################################
template<class T> void MPM_Example<T,3>::
Rasterize_Voxels()
{
    using Cell_Iterator             = Grid_Iterator_Cell<T,3>;
    using Hierarchy_Initializer     = Grid_Hierarchy_Initializer<Struct_type,T,3>;
    const Grid<T,3>& grid=hierarchy->Lattice(level);
    Range<int,3> bounding_grid_cells(grid.Clamp_To_Cell(bbox.min_corner),grid.Clamp_To_Cell(bbox.max_corner));
    for(Cell_Iterator iterator(grid,bounding_grid_cells);iterator.Valid();iterator.Next()){
        hierarchy->Activate_Cell(0,iterator.Cell_Index(),Cell_Type_Dirichlet);}
    
    hierarchy->Update_Block_Offsets();
    hierarchy->Initialize_Red_Black_Partition(2*threads);
}
//######################################################################
// Initialize_SPGrid
//######################################################################
template<class T> void MPM_Example<T,2>::
Initialize_SPGrid(const int& oid)
{
    Compute_Bounding_Box(bbox,oid);
    if(hierarchy!=nullptr) delete hierarchy;
    hierarchy=new Hierarchy(counts,domain,levels);
}
//######################################################################
// Initialize_SPGrid
//######################################################################
template<class T> void MPM_Example<T,3>::
Initialize_SPGrid(const int& oid)
{
    Compute_Bounding_Box(bbox,oid);
    if(hierarchy!=nullptr) delete hierarchy;
    hierarchy=new Hierarchy(counts,domain,levels);
}
//######################################################################
// Initialize
//######################################################################
template<class T> void MPM_Example<T,2>::
Initialize() 
{
    Initialize_Particles(Base::test_number);
    for(int oid=0;oid<objects.size();++oid){
        T_Object& obj=objects(oid);
        Populate_Simulated_Particles(oid);
        waiting_particles=simulated_particles;
        Initialize_SPGrid(oid);
        particle_bins.Resize(threads,threads);
        Update_Particle_Weights(oid);
        Group_Particles(oid);
        Rasterize_Voxels();
        Rasterize(oid);
        Process_Waiting_Particles(oid);
        Update_Constitutive_Model_State(oid);
        if(affine){
            const Grid<T,2>& grid=hierarchy->Lattice(level);
#pragma omp parallel for
            for(unsigned i=0;i<obj.particle_ids.size();++i){
                const int pid=obj.particle_ids[i]; 
                T_Particle &p=particles(pid);
                if(p.valid){
                    T_Mat& Dp_inv=p.Dp_inv;
                    const TV& p_V=p.V;
                    const TV& p_X0=p.X0;
                    Dp_inv=T_Mat();
                    for(T_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),p,true);iterator.Valid();iterator.Next()){
                        T_INDEX cell_index=iterator.Current_Cell(); auto data=cell_index._data;
                        T weight=iterator.Weight();
                        TV weight_gradient=iterator.Weight_Gradient();
                        if(weight>(T)0.){
                            TV cell_X=grid.Center(cell_index);
                            TV diff_X=cell_X-p_X0;
                            Dp_inv+=T_Mat::Outer_Product(diff_X,diff_X)*weight;
                        }
                    }
                    Dp_inv=Dp_inv.Inverse();
                }
            }
        }
        T total_Je=(T)0.;
        for(unsigned i=0;i<obj.particle_ids.size();++i){
            const int pid=obj.particle_ids[i]; 
            T_Particle &p=particles(pid);
            if(p.valid){
                const T Je=p.constitutive_model.Fe.Determinant();
                total_Je+=Je;
            }
        }
        average_Je_prev=total_Je/obj.particle_ids.size();
        obj.first_time=false;
        obj.to_update=false;
    }
}
//######################################################################
// Initialize
//######################################################################
template<class T> void MPM_Example<T,3>::
Initialize() 
{
    Initialize_Particles(Base::test_number);
    for(int oid=0;oid<objects.size();++oid){
        T_Object& obj=objects(oid);
        Populate_Simulated_Particles(oid);
        waiting_particles=simulated_particles;
        Initialize_SPGrid(oid);
        particle_bins.Resize(threads,threads);
        Update_Particle_Weights(oid);
        Group_Particles(oid);
        Rasterize_Voxels();
        Rasterize(oid);
        Process_Waiting_Particles(oid);
        Update_Constitutive_Model_State(oid);
        if(affine){
            const Grid<T,3>& grid=hierarchy->Lattice(level);
#pragma omp parallel for
            for(unsigned i=0;i<obj.particle_ids.size();++i){
                const int pid=obj.particle_ids[i]; 
                T_Particle &p=particles(pid);
                if(p.valid){
                    T_Mat& Dp_inv=p.Dp_inv;
                    const TV& p_V=p.V;
                    const TV& p_X0=p.X0;
                    Dp_inv=T_Mat();
                    for(T_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),p,true);iterator.Valid();iterator.Next()){
                        T_INDEX cell_index=iterator.Current_Cell(); auto data=cell_index._data;
                        T weight=iterator.Weight();
                        TV weight_gradient=iterator.Weight_Gradient();
                        if(weight>(T)0.){
                            TV cell_X=grid.Center(cell_index);
                            TV diff_X=cell_X-p_X0;
                            Dp_inv+=T_Mat::Outer_Product(diff_X,diff_X)*weight;
                        }
                    }
                    Dp_inv=Dp_inv.Inverse();
                }
            }
        }
        T total_Je=(T)0.;
        for(unsigned i=0;i<obj.particle_ids.size();++i){
            const int pid=obj.particle_ids[i]; 
            T_Particle &p=particles(pid);
            if(p.valid){
                const T Je=p.constitutive_model.Fe.Determinant();
                total_Je+=Je;
            }
        }
        average_Je_prev=total_Je/obj.particle_ids.size();
        obj.first_time=false;
        obj.to_update=false;
    }
}
//######################################################################
// Solve
//######################################################################
template<class T> T MPM_Example<T,2>::
Solve(const T& time) 
{
    high_resolution_clock::time_point tb_total=high_resolution_clock::now();
    
    high_resolution_clock::time_point tb_contact=high_resolution_clock::now();
    if(objects.size()>1) Apply_Contact_Force();
    high_resolution_clock::time_point te_contact=high_resolution_clock::now();
    duration<double> dur_contact=duration_cast<duration<double>>(te_contact-tb_contact);
    rt_contact+=dur_contact.count(); cnt_contact+=(T)1.;

    for(int oid=0;oid<objects.size();++oid){
        T_Object& obj=objects(oid);
        // obj.to_update=true;
        if(obj.to_update){
            const T_Object& obj=objects(oid);   
#pragma omp parallel for
            for(int i=0;i<obj.particle_ids.size();++i){
                const int pid=obj.particle_ids[i];
                T_Particle& p=particles(pid);
                // if(p.valid && (p.eos || (!p.eos && p.constitutive_model.plastic))) 
                if(p.valid) 
                    p.X0=p.X;
            }
        }
        high_resolution_clock::time_point tb_pop=high_resolution_clock::now();
        Populate_Simulated_Particles(oid);
        high_resolution_clock::time_point te_pop=high_resolution_clock::now();
        duration<double> dur_pop=duration_cast<duration<double>>(te_pop-tb_pop);
        rt_pop+=dur_pop.count(); cnt_pop+=(T)1.;        

        high_resolution_clock::time_point tb_init_spgrid=high_resolution_clock::now();
        Initialize_SPGrid(oid);
        high_resolution_clock::time_point te_init_spgrid=high_resolution_clock::now();
        duration<double> dur_init_spgrid=duration_cast<duration<double>>(te_init_spgrid-tb_init_spgrid);
        rt_init_spgrid+=dur_init_spgrid.count(); cnt_init_spgrid+=(T)1.;        

        particle_bins.Resize(threads,threads);

        high_resolution_clock::time_point tb_upw=high_resolution_clock::now();
        Update_Particle_Weights(oid);
        high_resolution_clock::time_point te_upw=high_resolution_clock::now();
        duration<double> dur_upw=duration_cast<duration<double>>(te_upw-tb_upw);
        rt_update_w+=dur_upw.count(); cnt_update_w+=(T)1.;    
        
        high_resolution_clock::time_point tb_g=high_resolution_clock::now();
        Group_Particles(oid);
        high_resolution_clock::time_point te_g=high_resolution_clock::now();
        duration<double> dur_g=duration_cast<duration<double>>(te_g-tb_g);
        rt_group+=dur_g.count(); cnt_group+=(T)1.; 

        high_resolution_clock::time_point tb_rv=high_resolution_clock::now();
        Rasterize_Voxels();   
        high_resolution_clock::time_point te_rv=high_resolution_clock::now();
        duration<double> dur_rv=duration_cast<duration<double>>(te_rv-tb_rv);
        rt_ras_vox+=dur_rv.count(); cnt_ras_vox+=(T)1.;

        high_resolution_clock::time_point tb_ras=high_resolution_clock::now();
        Rasterize(oid);
        high_resolution_clock::time_point te_ras=high_resolution_clock::now();
        duration<double> dur_ras=duration_cast<duration<double>>(te_ras-tb_ras);
        rt_ras+=dur_ras.count(); cnt_ras+=(T)1.;

        high_resolution_clock::time_point tb_ucms=high_resolution_clock::now();
        Update_Constitutive_Model_State(oid); // SVD, update Fe, Fp
        high_resolution_clock::time_point te_ucms=high_resolution_clock::now();
        duration<double> dur_ucms=duration_cast<duration<double>>(te_ucms-tb_ucms);
        rt_update_consti+=dur_ucms.count(); cnt_update_consti+=(T)1.;

        if(obj.to_update){
            const T_Object& obj=objects(oid);   
#pragma omp parallel for
            for(int i=0;i<obj.particle_ids.size();++i){
                const int pid=obj.particle_ids[i];
                T_Particle& p=particles(pid);
                // if(p.valid && (p.eos || (!p.eos && p.constitutive_model.plastic))){
                if(p.valid){
                    p.constitutive_model.Fes=T_Mat::Identity_Matrix();  // Fe_sn
                    p.constitutive_model.Fes0=p.constitutive_model.Fe;  // Fe_0s
                }
            }
        }

        Update_Particle_Velocities_And_Positions(oid); // update Fe (n+1)
        
        obj.to_update=false;
    #if 1
        T total_Je=(T)0.;
        int cnt=0;
        T minJe=1e5; T maxJe=-1e5;
        T minRho=1e5; T maxRho=-1e5;
        for(unsigned i=0;i<obj.particle_ids.size();++i){
            const int pid=obj.particle_ids[i]; 
            T_Particle &p=particles(pid);
            if(p.valid){
                const T Je=p.constitutive_model.Fes.Determinant(); 
                const T rho=p.density;
                if(rho>maxRho) maxRho=rho;
                else if(rho<minRho) minRho=rho;

                if(Je>maxJe) maxJe=Je;
                else if(Je<minJe) minJe=Je;
                const T diff=std::abs(Je-(T)1.);
                if(diff>percentage) cnt++;
                const T criteria=p.eos?(T).05:(T).75;
                if(cnt>obj.particle_ids.size()*criteria){ // liquid
                    obj.to_update=true;
                    break;
                }
            }
        }
        Log::cout<<"#############################"<<std::endl;
        Log::cout<<"cnt/total: "<<(T)100.*cnt/obj.particle_ids.size()<<"%"<<std::endl;
        Log::cout<<"Je range: "<<minJe<<", "<<maxJe<<std::endl;
        Log::cout<<"Rho range: "<<minRho<<", "<<maxRho<<std::endl;
        Log::cout<<"#############################"<<std::endl;
    #endif
    }
    high_resolution_clock::time_point te_total=high_resolution_clock::now();
    duration<double> dur_total=duration_cast<duration<double>>(te_total-tb_total);
    rt_total+=dur_total.count(); cnt_total+=(T)1.;
    return dt;
}
//######################################################################
// Solve
//######################################################################
template<class T> T MPM_Example<T,3>::
Solve(const T& time)
{
    Spiral_Test(time,dt);
#if 1
    if(add_objects) Add_Object();
    if(objects.size()>1) Apply_Contact_Force();
    for(int oid=0;oid<objects.size();++oid){
        T_Object& obj=objects(oid);
        high_resolution_clock::time_point tb1=high_resolution_clock::now();
        if(obj.to_update){
            const T_Object& obj=objects(oid);   
#pragma omp parallel for
            for(int i=0;i<obj.particle_ids.size();++i){
                const int pid=obj.particle_ids[i];
                T_Particle& p=particles(pid);
                if(p.valid) p.X0=p.X;
            }
	        update_cnt++;
        }
        high_resolution_clock::time_point te1=high_resolution_clock::now();
	    duration<double> dur1=duration_cast<duration<double>>(te1-tb1);
        update_rt+=dur1.count();

        Populate_Simulated_Particles(oid);
        Initialize_SPGrid(oid);
        particle_bins.Resize(threads,threads);
        Update_Particle_Weights(oid);
        Group_Particles(oid);
        Rasterize_Voxels();        
        Rasterize(oid);
        Update_Constitutive_Model_State(oid); // SVD, update Fe, Fp
        high_resolution_clock::time_point tb2=high_resolution_clock::now();
        if(obj.to_update){
            const T_Object& obj=objects(oid);   
#pragma omp parallel for
            for(int i=0;i<obj.particle_ids.size();++i){
                const int pid=obj.particle_ids[i];
                T_Particle& p=particles(pid);
                if(p.valid){
                    p.constitutive_model.Fes=T_Mat::Identity_Matrix();  // Fe_sn
                    p.constitutive_model.Fes0=p.constitutive_model.Fe;  // Fe_0s
                }
            }
        }
        high_resolution_clock::time_point te2=high_resolution_clock::now();
	    duration<double> dur2=duration_cast<duration<double>>(te2-tb2);
        update_rt+=dur2.count();
        Update_Particle_Velocities_And_Positions(oid); // update Fe (n+1)
        obj.to_update=false;
    #if 1
        T total_Je=(T)0.;
        int cnt=0;
        T minJe=1e5; T maxJe=-1e5;
        T minRho=1e5; T maxRho=-1e5;
        for(unsigned i=0;i<obj.particle_ids.size();++i){
            const int pid=obj.particle_ids[i]; 
            T_Particle &p=particles(pid);
            if(p.valid){
                const T Je=p.constitutive_model.Fes.Determinant(); 
                const T rho=p.density;
                if(rho>maxRho) maxRho=rho;
                else if(rho<minRho) minRho=rho;

                if(Je>maxJe) maxJe=Je;
                else if(Je<minJe) minJe=Je;
                const T diff=std::abs(Je-(T)1.);
                if(diff>percentage) cnt++;
                const T criteria=p.eos?(T).01:(T).1;
                if(cnt>obj.particle_ids.size()*criteria){ 
                    obj.to_update=true;
                    break;
                }
            }
        }
        Log::cout<<"#############################"<<std::endl;
        Log::cout<<"cnt/total: "<<(T)100.*cnt/obj.particle_ids.size()<<"%"<<std::endl;
        Log::cout<<"Je range: "<<minJe<<", "<<maxJe<<std::endl;
        Log::cout<<"Rho range: "<<minRho<<", "<<maxRho<<std::endl;
        Log::cout<<"#############################"<<std::endl;
    #endif
    }
    Log::cout<<"Frame: "<<Base::current_frame<<std::endl;
    Log::cout<<"total update rt: "<<update_rt<<std::endl;
    Log::cout<<"total update cnt: "<<update_cnt<<std::endl;
#endif
    return dt;
}
//######################################################################
// Rasterize
//######################################################################
template<class T> void MPM_Example<T,2>::
Rasterize(const int& oid) 
{
    auto flags=hierarchy->Channel(level,flags_channel);             
    auto mass=hierarchy->Channel(level,mass_channel);               
    auto x0=hierarchy->Channel(level,position_channels(0));         auto x1=hierarchy->Channel(level,position_channels(1));
    auto v0=hierarchy->Channel(level,velocity_channels(0));         auto v1=hierarchy->Channel(level,velocity_channels(1));
    auto bv0=hierarchy->Channel(level,barrier_velocity_channels(0));         
    auto bv1=hierarchy->Channel(level,barrier_velocity_channels(1));


    auto X0_x=hierarchy->Channel(level,X0_channels(0));             auto X0_y=hierarchy->Channel(level,X0_channels(1));         

    const Grid<T,2>& grid=hierarchy->Lattice(level);
    T_Object& obj=objects(oid);
#pragma omp parallel for
    for(int tid_process=0;tid_process<threads;++tid_process){
        const Interval<int> thread_x_interval=x_intervals(tid_process);
        for(int tid_collect=0;tid_collect<threads;++tid_collect){
            Array<int>& index=particle_bins(tid_process,tid_collect);
            for(int i=0;i<index.size();++i){
                T_Particle& p=particles(index(i));T_INDEX& closest_cell=p.closest_cell;
                const Interval<int> relative_interval=Interval<int>(thread_x_interval.min_corner-closest_cell(0),thread_x_interval.max_corner-closest_cell(0));
                for(T_Cropped_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),relative_interval,p,true);iterator.Valid();iterator.Next()){
                    T weight=iterator.Weight();
                    TV p_X=p.X; TV p_X0=p.X0; TV p_V=p.V; T p_mass=p.mass; 
                    if(affine) p_V+=p.Cv*(grid.Center(iterator.Current_Cell())-p.X0);
                    auto data=iterator.Current_Cell()._data;
                    x0(data)+=weight*(p_mass*p_X(0));       x1(data)+=weight*(p_mass*p_X(1));
                    X0_x(data)+=weight*(p_mass*p_X0(0));    X0_y(data)+=weight*(p_mass*p_X0(1));
                    v0(data)+=weight*(p_mass*p_V(0));       v1(data)+=weight*(p_mass*p_V(1));
                    mass(data)+=weight*p_mass; 
                }
            }
        }
    }
    // set flags
    Flag_Setup_Helper<Struct_type,T,2>(hierarchy->Allocator(level),hierarchy->Blocks(level),mass_channel);        
    Vector_Normalization_Helper<Struct_type,T,2>(hierarchy->Allocator(level),hierarchy->Blocks(level),mass_channel,velocity_channels);
    Vector_Normalization_Helper<Struct_type,T,2>(hierarchy->Allocator(level),hierarchy->Blocks(level),mass_channel,position_channels);
        
    Vector_Normalization_Helper<Struct_type,T,2>(hierarchy->Allocator(level),hierarchy->Blocks(level),mass_channel,X0_channels);
    if(objects.size()>1){
        int cnt=0;
        for(int bid=0;bid<obj.barriers.size();++bid){
            T_INDEX barrier_index=obj.barriers(bid);
            auto data=barrier_index._data;
            flags(data)|=Cell_Type_Solid;
            TV bV=obj.velocities(bid);
            bv0(data)=bV(0); bv1(data)=bV(1);
        }
    }
}
//######################################################################
// Rasterize
//######################################################################
template<class T> void MPM_Example<T,3>::
Rasterize(const int& oid) 
{
    auto flags=hierarchy->Channel(level,flags_channel);             
    auto mass=hierarchy->Channel(level,mass_channel);               
    auto x0=hierarchy->Channel(level,position_channels(0));         
    auto x1=hierarchy->Channel(level,position_channels(1));
    auto x2=hierarchy->Channel(level,position_channels(2));
    auto v0=hierarchy->Channel(level,velocity_channels(0));         
    auto v1=hierarchy->Channel(level,velocity_channels(1));
    auto v2=hierarchy->Channel(level,velocity_channels(2));
    auto bv0=hierarchy->Channel(level,barrier_velocity_channels(0));         
    auto bv1=hierarchy->Channel(level,barrier_velocity_channels(1));
    auto bv2=hierarchy->Channel(level,barrier_velocity_channels(2));

    const Grid<T,3>& grid=hierarchy->Lattice(level);
    T_Object& obj=objects(oid);
#pragma omp parallel for
    for(int tid_process=0;tid_process<threads;++tid_process){
        const Interval<int> thread_x_interval=x_intervals(tid_process);
        for(int tid_collect=0;tid_collect<threads;++tid_collect){
            Array<int>& index=particle_bins(tid_process,tid_collect);
            for(int i=0;i<index.size();++i){
                T_Particle& p=particles(index(i));T_INDEX& closest_cell=p.closest_cell;
                const Interval<int> relative_interval=Interval<int>(thread_x_interval.min_corner-closest_cell(0),thread_x_interval.max_corner-closest_cell(0));
                for(T_Cropped_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),relative_interval,p,true);iterator.Valid();iterator.Next()){
                    T weight=iterator.Weight();
                    TV p_X=p.X; TV p_V=p.V; T p_mass=p.mass; 
                    if(affine) p_V+=p.Cv*(grid.Center(iterator.Current_Cell())-p.X0);
                    auto data=iterator.Current_Cell()._data;
                    x0(data)+=weight*(p_mass*p_X(0)); x1(data)+=weight*(p_mass*p_X(1)); x2(data)+=weight*(p_mass*p_X(2));
                    v0(data)+=weight*(p_mass*p_V(0)); v1(data)+=weight*(p_mass*p_V(1)); v2(data)+=weight*(p_mass*p_V(2)); 
                    mass(data)+=weight*p_mass; 
                }
            }
        }
    }
    // set flags
    Flag_Setup_Helper<Struct_type,T,3>(hierarchy->Allocator(level),hierarchy->Blocks(level),mass_channel);        
    Vector_Normalization_Helper<Struct_type,T,3>(hierarchy->Allocator(level),hierarchy->Blocks(level),mass_channel,velocity_channels);
    Vector_Normalization_Helper<Struct_type,T,3>(hierarchy->Allocator(level),hierarchy->Blocks(level),mass_channel,position_channels);
        
    if(objects.size()>1){
        int cnt=0;
        for(int bid=0;bid<obj.barriers.size();++bid){
            T_INDEX barrier_index=obj.barriers(bid);
            auto data=barrier_index._data;
            flags(data)|=Cell_Type_Solid;
            TV bV=obj.velocities(bid);
            bv0(data)=bV(0); bv1(data)=bV(1); bv2(data)=bV(2); 
        }
    }
}
//######################################################################
// Process_Waiting_Particles
//######################################################################
template<class T> void MPM_Example<T,2>::
Process_Waiting_Particles(const int& oid)
{   
    auto mass=hierarchy->Channel(level,mass_channel);
    const Grid<T,2>& grid=hierarchy->Lattice(level); const T one_over_volume_per_cell=(T)1./grid.dX.Product();
#pragma omp parallel for
    for(unsigned i=0;i<waiting_particles.size();++i){
        const int id=waiting_particles(i); T_Particle& p=particles(id); 
        T particle_density=(T)0.;
        if(p.valid){
            for(T_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),p,true);iterator.Valid();iterator.Next()){
                T_INDEX relative_index=iterator.Index()+T_INDEX(1);
                T weight=iterator.Weight();
                particle_density+=weight*mass(iterator.Current_Cell()._data);}
            particle_density*=one_over_volume_per_cell;
            p.volume=p.mass/particle_density;
        }
    }
    waiting_particles.Clear();
}
//######################################################################
// Process_Waiting_Particles
//######################################################################
template<class T> void MPM_Example<T,3>::
Process_Waiting_Particles(const int& oid)
{   
    auto mass=hierarchy->Channel(level,mass_channel);
    const Grid<T,3>& grid=hierarchy->Lattice(level); const T one_over_volume_per_cell=(T)1./grid.dX.Product();
#pragma omp parallel for
    for(unsigned i=0;i<waiting_particles.size();++i){
        const int id=waiting_particles(i); T_Particle& p=particles(id); 
        T particle_density=(T)0.;
        if(p.valid){
            for(T_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),p,true);iterator.Valid();iterator.Next()){
                T_INDEX relative_index=iterator.Index()+T_INDEX(1);
                T weight=iterator.Weight();
                particle_density+=weight*mass(iterator.Current_Cell()._data);}
            particle_density*=one_over_volume_per_cell;
            p.volume=p.mass/particle_density;
        }
    }
    waiting_particles.Clear();
}
//######################################################################
// Update_Particle_Velocities_And_Positions
//######################################################################
template<class T> void MPM_Example<T,2>::
Update_Particle_Velocities_And_Positions(const int& oid)
{
    Array<Array<int> > remove_indices(threads);
    auto vs0=hierarchy->Channel(level,velocity_star_channels(0));   auto vs1=hierarchy->Channel(level,velocity_star_channels(1));
    auto v0=hierarchy->Channel(level,velocity_channels(0));         auto v1=hierarchy->Channel(level,velocity_channels(1)); 
    auto mass=hierarchy->Channel(level,mass_channel);

    Apply_Force(oid);
    
    const Grid<T,2>& grid=hierarchy->Lattice(level);
    T_Object& obj=objects(oid);
    Range<T,2> cell_domain(grid.domain.min_corner+(T).5*grid.dX,grid.domain.max_corner-(T).5*grid.dX);

#if 0
    // compute Dp
    if(affine&(obj.first_time||obj.to_update)){
#pragma omp parallel for
        for(unsigned i=0;i<obj.particle_ids.size();++i){
            const int pid=obj.particle_ids[i]; 
            T_Particle &p=particles(pid);
            if(p.valid && (p.eos || (!p.eos && p.constitutive_model.plastic))){
            // if(p.valid){
                T_Mat& Dp_inv=p.Dp_inv;
                const TV& p_V=p.V;
                const TV& p_X0=p.X0;
                Dp_inv=T_Mat();
                for(T_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),p,true);iterator.Valid();iterator.Next()){
                    T_INDEX cell_index=iterator.Current_Cell(); auto data=cell_index._data;
                    T weight=iterator.Weight();
                    TV weight_gradient=iterator.Weight_Gradient();
                    if(weight>(T)0.){
                        TV cell_X=grid.Center(cell_index);
                        TV diff_X=cell_X-p_X0;
                        Dp_inv+=T_Mat::Outer_Product(diff_X,diff_X)*weight; 
                        // (dx dy)^T * (dx dy) = 
                        //  (dx^2   dx*dy) * weight  =>  \sum weight*dx^2 = cell_width^2/4
                        //  (dx*dy   dy^2)
                    }
                }
                Dp_inv=Dp_inv.Inverse();
                if(false){
                    Log::cout<<"Dp_inv: "<<Dp_inv<<std::endl;
                    Log::cout<<(T)4.*Nova_Utilities::Sqr(grid.one_over_dX(0))<<std::endl;
                }
            }
        }
    }
#endif
    high_resolution_clock::time_point tb_g2p=high_resolution_clock::now();
    Array<int> invalid_indices_in_obj;
#pragma omp parallel for
    for(unsigned i=0;i<obj.particle_ids.size();++i){
        const int pid=obj.particle_ids[i]; 
        T_Particle &p=particles(pid);
        if(p.valid){
        TV V_pic=TV(),V_flip=p.V; 
        T_Mat grad_Vps=T_Mat();
        T_Mat C_pic;

        for(T_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),p,true);iterator.Valid();iterator.Next()){
            T_INDEX relative_index=iterator.Index()+T_INDEX(1);
            T_INDEX cell_index=iterator.Current_Cell();
            auto data=iterator.Current_Cell()._data;
            T weight=iterator.Weight();
            TV weight_gradient=iterator.Weight_Gradient();
            
            if(weight>(T)0.){
                TV V_grid({vs0(data),vs1(data)}),delta_V_grid({vs0(data)-v0(data),vs1(data)-v1(data)});
                V_pic+=weight*V_grid; 
                V_flip+=weight*delta_V_grid;
            }}

            if(affine){
                flip=(T)0.;
            }
            
            if(!p.fixed) {
                p.V=V_flip*flip+V_pic*((T)1.-flip);
                if(affine) p.X+=p.V*dt;
                else p.X+=V_pic*dt;
            }
            else {
                p.V=p.fixed_velocity;
                p.X+=p.V*dt;
            }

        // update F and density
        const TV& p_V=p.V;
        const TV& p_X0=p.X0;
        const T_Mat& Dp_inv=p.Dp_inv;
        for(T_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),p,true);iterator.Valid();iterator.Next()){
            T_INDEX relative_index=iterator.Index()+T_INDEX(1);
            T_INDEX cell_index=iterator.Current_Cell();
            auto data=iterator.Current_Cell()._data;
            T weight=iterator.Weight();
            TV weight_gradient=iterator.Weight_Gradient();
            if(weight>(T)0.){
                if(affine){
                    TV cell_V=TV({vs0(data),vs1(data)});
                    TV cell_X=grid.Center(cell_index);
                    TV diffV=cell_V-(T)0.*p_V;
                    TV diffX=cell_X-p_X0;
                    if(use_inv) grad_Vps+=weight*T_Mat::Outer_Product(diffV,diffX)*Dp_inv;
                    else grad_Vps+=weight*T_Mat::Outer_Product(diffV,diffX)*(T)4*Nova_Utilities::Sqr(grid.one_over_dX(0));
                }
                else{
                    TV V_grid({vs0(data),vs1(data)});
                    grad_Vps+=T_Mat::Outer_Product(V_grid,weight_gradient);
                }
            }
        }
        p.density/=(T)1.+dt*(grad_Vps*(p.constitutive_model.Fes.Inverse())).Trace();
        p.constitutive_model.Fe_prev=p.constitutive_model.Fe;
        if(!p.constitutive_model.plastic) {
            if(!p.eos) p.constitutive_model.Fe=(p.constitutive_model.Fes+dt*grad_Vps)*p.constitutive_model.Fes0; // Fe_(0,n+1)
            else {
                p.constitutive_model.Fe=std::sqrt((T)1.+dt*(grad_Vps*p.constitutive_model.Fes.Inverse()).Trace())*p.constitutive_model.Fes*p.constitutive_model.Fes0; // Fe_(0,n+1)
            }
        }
        if(!p.eos) p.constitutive_model.Fes+=dt*grad_Vps;   // Fe_(s,n+1)      
        else p.constitutive_model.Fes=std::sqrt((T)1.+dt*(grad_Vps*p.constitutive_model.Fes.Inverse()).Trace())*p.constitutive_model.Fes;   // Fe_(s,n+1)      
        if(p.constitutive_model.plastic) p.constitutive_model.Fe+=dt*grad_Vps*p.constitutive_model.Fes.Inverse()*p.constitutive_model.Fe; // Fe_(0,n+1)

        
        // if(std::abs(p.constitutive_model.Fe.Determinant()-(T)1.)>(T)1.e-5) Log::cout<<p.constitutive_model.Fe<<std::endl;
        if(affine){
            p.Cv=grad_Vps;
        }
        if(!cell_domain.Inside(p.X)){
            p.valid=false;
        }
        }
    }
    high_resolution_clock::time_point te_g2p=high_resolution_clock::now();
    duration<double> dur_g2p=duration_cast<duration<double>>(te_g2p-tb_g2p);
    rt_g2p+=dur_g2p.count(); cnt_g2p+=(T)1.;
}
//######################################################################
// Update_Particle_Velocities_And_Positions
//######################################################################
template<class T> void MPM_Example<T,3>::
Update_Particle_Velocities_And_Positions(const int& oid)
{
    Array<Array<int> > remove_indices(threads);
    auto vs0=hierarchy->Channel(level,velocity_star_channels(0));   
    auto vs1=hierarchy->Channel(level,velocity_star_channels(1));
    auto vs2=hierarchy->Channel(level,velocity_star_channels(2));
    auto v0=hierarchy->Channel(level,velocity_channels(0));         
    auto v1=hierarchy->Channel(level,velocity_channels(1)); 
    auto v2=hierarchy->Channel(level,velocity_channels(2)); 
    auto mass=hierarchy->Channel(level,mass_channel);  
    Apply_Force(oid);
    const Grid<T,3>& grid=hierarchy->Lattice(level);
    T_Object& obj=objects(oid);
    Range<T,3> cell_domain(grid.domain.min_corner+(T).5*grid.dX,grid.domain.max_corner-(T).5*grid.dX);

#if 0
    // compute Dp
    if(affine&(obj.first_time||obj.to_update)){
#pragma omp parallel for
        for(unsigned i=0;i<obj.particle_ids.size();++i){
            const int pid=obj.particle_ids[i]; 
            T_Particle &p=particles(pid);
            // if(p.valid && (p.eos || (!p.eos && p.constitutive_model.plastic))){
            if(p.valid){
                T_Mat& Dp_inv=p.Dp_inv;
                const TV& p_V=p.V;
                const TV& p_X0=p.X0;
                Dp_inv=T_Mat();
                for(T_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),p,true);iterator.Valid();iterator.Next()){
                    T_INDEX cell_index=iterator.Current_Cell(); auto data=cell_index._data;
                    T weight=iterator.Weight();
                    TV weight_gradient=iterator.Weight_Gradient();
                    if(weight>(T)0.){
                        TV cell_X=grid.Center(cell_index);
                        TV diff_X=cell_X-p_X0;
                        Dp_inv+=T_Mat::Outer_Product(diff_X,diff_X)*weight;
                    }
                }
                Dp_inv=Dp_inv.Inverse();
            }
        }
    }
#endif
    Array<int> invalid_indices_in_obj;
#pragma omp parallel for
    for(unsigned i=0;i<obj.particle_ids.size();++i){
        const int pid=obj.particle_ids[i]; 
        T_Particle &p=particles(pid);
        if(p.valid){
            TV V_pic=TV(),V_flip=p.V; 
            T_Mat grad_Vps=T_Mat();
            T_Mat C_pic;

            for(T_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),p,true);iterator.Valid();iterator.Next()){
                T_INDEX relative_index=iterator.Index()+T_INDEX(1);
                T_INDEX cell_index=iterator.Current_Cell();
                auto data=iterator.Current_Cell()._data;
                T weight=iterator.Weight();
                TV weight_gradient=iterator.Weight_Gradient();
            
                if(weight>(T)0.){
                    TV V_grid({vs0(data),vs1(data),vs2(data)}),delta_V_grid({vs0(data)-v0(data),vs1(data)-v1(data),vs2(data)-v2(data)});
                    V_pic+=weight*V_grid; 
                    V_flip+=weight*delta_V_grid;
                }
            }

            if(affine) flip=(T)0.;
            
            if(!p.fixed) {
                p.V=V_flip*flip+V_pic*((T)1.-flip);
                if(affine) p.X+=p.V*dt;
                else p.X+=V_pic*dt;
            }
            else {
                p.V=p.fixed_velocity;
                p.X+=p.V*dt;
            }

            // update F and density
            const TV& p_V=p.V;
            const TV& p_X0=p.X0;
            const T_Mat& Dp_inv=p.Dp_inv;
            for(T_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),p,true);iterator.Valid();iterator.Next()){
                T_INDEX relative_index=iterator.Index()+T_INDEX(1);
                T_INDEX cell_index=iterator.Current_Cell();
                auto data=iterator.Current_Cell()._data;
                T weight=iterator.Weight();
                TV weight_gradient=iterator.Weight_Gradient();
                if(weight>(T)0.){
                    if(affine){
                        TV cell_V=TV({vs0(data),vs1(data),vs2(data)});
                        TV cell_X=grid.Center(cell_index);
                        TV diffV=cell_V-p_V;
                        TV diffX=cell_X-p_X0;
                        if(use_inv) grad_Vps+=weight*T_Mat::Outer_Product(diffV,diffX)*Dp_inv;
                        else grad_Vps+=weight*T_Mat::Outer_Product(diffV,diffX)*(T)4*Nova_Utilities::Sqr(grid.one_over_dX(0));
                    }
                    else{
                        TV V_grid({vs0(data),vs1(data),vs2(data)});
                        grad_Vps+=T_Mat::Outer_Product(V_grid,weight_gradient);
                    }
                }
            }
            p.density/=(T)1.+dt*(grad_Vps*(p.constitutive_model.Fes.Inverse())).Trace();

            p.constitutive_model.Fe_prev=p.constitutive_model.Fe;
            if(!p.constitutive_model.plastic) p.constitutive_model.Fe=(p.constitutive_model.Fes+dt*grad_Vps)*p.constitutive_model.Fes0; // Fe_(0,n+1)
            p.constitutive_model.Fes+=dt*grad_Vps;   // Fe_(s,n+1)      
            if(p.constitutive_model.plastic) p.constitutive_model.Fe+=dt*grad_Vps*p.constitutive_model.Fes.Inverse()*p.constitutive_model.Fe; // Fe_(0,n    +1)

        
            // if(std::abs(p.constitutive_model.Fe.Determinant()-(T)1.)>(T)1.e-5) Log::cout<<p.constitutive_model.Fe<<std::endl;
            if(affine){
                p.Cv=grad_Vps;
            }
            if(!cell_domain.Inside(p.X)){
                p.valid=false;
            }
        }
    }
}
//######################################################################
// Apply_Force
//######################################################################
template<class T> void MPM_Example<T,2>::
Apply_Force(const int& oid) 
{
    high_resolution_clock::time_point tb_exp=high_resolution_clock::now();
    Apply_Explicit_Force(oid);
    high_resolution_clock::time_point te_exp=high_resolution_clock::now();
    duration<double> dur_exp=duration_cast<duration<double>>(te_exp-tb_exp);
    rt_exp_force+=dur_exp.count(); cnt_exp_force+=(T)1.;

    high_resolution_clock::time_point tb_b=high_resolution_clock::now();
    Grid_Based_Collision(oid);
    high_resolution_clock::time_point te_b=high_resolution_clock::now();
    duration<double> dur_b=duration_cast<duration<double>>(te_b-tb_b);
    rt_boundary+=dur_b.count(); cnt_boundary+=(T)1.;
    
    high_resolution_clock::time_point tb_im=high_resolution_clock::now();
    if(implicit_solve){
        const T solver_tolerance=1.e-7;
        const int solver_iterations=10000;
        T_Object& obj=objects(oid);
        Conjugate_Gradient<T> cg;
        Krylov_Solver<T>* solver=(Krylov_Solver<T>*)&cg;
        Array<Rigid_Cup<T,2> > cups;
        MPM_CG_System<Struct_type,T,2> mpm_system(*hierarchy,obj.particle_ids,particles,particle_bins,x_intervals,barriers,cups,fixed_domains,position_channels,(T)0.,dt,threads);
        mpm_system.use_preconditioner=false;
        // set rhs here
        MPM_RHS_Helper<Struct_type,T,2>(hierarchy->Allocator(level),hierarchy->Blocks(level),velocity_star_channels,rhs_channels);     
        MPM_CG_Vector<Struct_type,T,2> solver_vp(*hierarchy,velocity_star_channels),solver_rhs(*hierarchy,rhs_channels),solver_q(*hierarchy,q_channels),solver_s(*hierarchy,s_channels),solver_r(*hierarchy,r_channels),solver_k(*hierarchy,z_channels),solver_z(*hierarchy,z_channels);
        solver->Solve(mpm_system,solver_vp,solver_rhs,solver_q,solver_s,solver_r,solver_k,solver_z,solver_tolerance,0,solver_iterations);
    }
    high_resolution_clock::time_point te_im=high_resolution_clock::now();
    duration<double> dur_im=duration_cast<duration<double>>(te_im-tb_im);
    rt_implicit+=dur_im.count(); cnt_implicit+=(T)1.;
}
//######################################################################
// Apply_Force
//######################################################################
template<class T> void MPM_Example<T,3>::
Apply_Force(const int& oid) 
{
    Apply_Viscous_Force(oid);
    Apply_Explicit_Force(oid);
    Grid_Based_Collision(oid);
    if(implicit_solve){
        const T solver_tolerance=1.e-7;
        const int solver_iterations=10000;
        T_Object& obj=objects(oid);
        Conjugate_Gradient<T> cg;
        Krylov_Solver<T>* solver=(Krylov_Solver<T>*)&cg;
        MPM_CG_System<Struct_type,T,3> mpm_system(*hierarchy,obj.particle_ids,particles,particle_bins,x_intervals,barriers,cups,fixed_domains,position_channels,(T)0.,dt,threads);
        mpm_system.use_preconditioner=false;
        // set rhs here
        MPM_RHS_Helper<Struct_type,T,3>(hierarchy->Allocator(level),hierarchy->Blocks(level),velocity_star_channels,rhs_channels);     
        MPM_CG_Vector<Struct_type,T,3> solver_vp(*hierarchy,velocity_star_channels),solver_rhs(*hierarchy,rhs_channels),solver_q(*hierarchy,q_channels),solver_s(*hierarchy,s_channels),solver_r(*hierarchy,r_channels),solver_k(*hierarchy,z_channels),solver_z(*hierarchy,z_channels);
        solver->Solve(mpm_system,solver_vp,solver_rhs,solver_q,solver_s,solver_r,solver_k,solver_z,solver_tolerance,0,solver_iterations);
    }
}
//######################################################################
// Apply_Viscous_Force
//######################################################################
template<class T> void MPM_Example<T,3>::
Apply_Viscous_Force(const int& oid)
{
    const Grid<T,3>& grid=hierarchy->Lattice(level);
    auto f0=hierarchy->Channel(level,f_channels(0));        
    auto f1=hierarchy->Channel(level,f_channels(1));
    auto f2=hierarchy->Channel(level,f_channels(2));
    auto v0=hierarchy->Channel(level,velocity_channels(0));        
    auto v1=hierarchy->Channel(level,velocity_channels(1));
    auto v2=hierarchy->Channel(level,velocity_channels(2));
    Array<int> thread_cnt(threads);

    const T one_over_dt2=(T)1/sqr(dt);
    T_Object& obj=objects(oid);
// #pragma omp parallel for
    for(int tid_process=0;tid_process<threads;++tid_process){
        const Interval<int>& thread_x_interval=x_intervals(tid_process);
        int& cnt=thread_cnt(tid_process);
        for(int tid_collect=0;tid_collect<threads;++tid_collect){
            const Array<int>& index=particle_bins(tid_process,tid_collect);
            for(int i=0;i<index.size();++i){
                T_Particle& p=particles(index(i)); T_INDEX& closest_cell=p.closest_cell;
                T eos_coefficient=(T)0.; T_Mat P,V0_P_FT;
                const T_Mat& Dp_inv=p.Dp_inv;
                const T_Mat& Fes=p.constitutive_model.Fes;
                const T_Mat& Fes0=p.constitutive_model.Fes0;
                const T_Mat& Fe=p.constitutive_model.Fe;
                const T Je=Fe.Determinant(); 
                const T Jes=Fes.Determinant();
                const T Jes0=Fes0.Determinant();
                const T V0=p.volume;
                eos_coefficient=V0/p.density*p.bulk_modulus*(pow(p.density,p.gamma)-(T)1);  // == sigma * V0 
                if(std::abs(Jes)<1.e-5) V0_P_FT=T_Mat();
                else V0_P_FT=Jes*eos_coefficient*(Fes.Inverse().Transposed());
                const T_Mat& F=p.constitutive_model.Fe;
                T_Mat A;
                for(T_Influence_Iterator it(T_INDEX(-1),T_INDEX(1),p,true);it.Valid();it.Next()){
                    auto data=it.Current_Cell()._data; TV vel=TV({v0(data),v1(data),v2(data)});
                    A+=T_Mat::Outer_Product(vel*dt,it.Weight_Gradient());
                }
                for(T_Influence_Iterator it(T_INDEX(-1),T_INDEX(1),p,true);it.Valid();it.Next()){
                    auto data=it.Current_Cell()._data;
                    TV qpi=(A*it.Weight_Gradient()+A.Transpose_Times(it.Weight_Gradient()))*one_over_dt2;
                    TV cpi=qpi*(p.viscosity*p.constitutive_model.Fe.Determinant());
                    TV rhs=V0*cpi;
                    f0(data)-=rhs(0);
                    f1(data)-=rhs(1);
                    f2(data)-=rhs(2);
                }
                // const Interval<int> relative_interval=Interval<int>(thread_x_interval.min_corner-closest_cell(0),thread_x_interval.max_corner-closest_cell(0));  

                // for(T_Cropped_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),relative_interval,p,true);iterator.Valid();iterator.Next()){
                //     T_INDEX relative_index=iterator.Index()+T_INDEX(1);
				// 	T weight=iterator.Weight();
                //     TV weight_gradient=iterator.Weight_Gradient();
                //     auto data=iterator.Current_Cell()._data;         
                //     TV pressure_force;
                //     if(affine){
                //         T_INDEX cell_index=iterator.Current_Cell();
                //         TV cell_X=grid.Center(cell_index);
                //         TV diffX=cell_X-p.X0;
                //         pressure_force=weight*V0_P_FT*diffX*(T)4.*Nova_Utilities::Sqr(grid.one_over_dX(0));
                //     }
                //     else{
                //         pressure_force=V0_P_FT*weight_gradient;
                //     }
                //     f0(data)+=pressure_force(0); 
                //     f1(data)+=pressure_force(1); 
                //     f2(data)+=pressure_force(2); 
            }
        }
    }
}
//######################################################################
// Apply_Explicit_Force
//######################################################################
template<class T> void MPM_Example<T,2>::
Apply_Explicit_Force(const int& oid)
{
    const Grid<T,2>& grid=hierarchy->Lattice(level);
    auto f0=hierarchy->Channel(level,f_channels(0));        auto f1=hierarchy->Channel(level,f_channels(1));
    Array<int> thread_cnt(threads);

    T_Object& obj=objects(oid);
#pragma omp parallel for
    for(int tid_process=0;tid_process<threads;++tid_process){
        const Interval<int>& thread_x_interval=x_intervals(tid_process);
        int& cnt=thread_cnt(tid_process);
        for(int tid_collect=0;tid_collect<threads;++tid_collect){
            const Array<int>& index=particle_bins(tid_process,tid_collect);
            for(int i=0;i<index.size();++i){
                T_Particle& p=particles(index(i)); T_INDEX& closest_cell=p.closest_cell;
                T eos_coefficient=(T)0.; T V0=p.volume; T_Mat P,F,V0_P_FT;
                const T_Mat& Dp_inv=p.Dp_inv;
                if(p.eos) {
                    const T_Mat& Fes=p.constitutive_model.Fes;
                    const T_Mat& Fes0=p.constitutive_model.Fes0;
                    const T_Mat& Fe=p.constitutive_model.Fe;
                    const T Je=Fe.Determinant(); 
                    const T Jes=Fes.Determinant();
                    const T Jes0=Fes0.Determinant();
                    eos_coefficient=V0/p.density*p.bulk_modulus*(pow(p.density,p.gamma)-(T)1);  // == sigma * V0 
                    if(std::abs(Jes)<1.e-5) V0_P_FT=T_Mat();
                    else V0_P_FT=Jes*eos_coefficient*(Fes.Inverse().Transposed());
                }
                else{ P=p.constitutive_model.P(),F=p.constitutive_model.Fes0; // sigma = 1/J * P * FT * V0 * J 
                    V0_P_FT=P.Times_Transpose(F)*V0;}
                const Interval<int> relative_interval=Interval<int>(thread_x_interval.min_corner-closest_cell(0),thread_x_interval.max_corner-closest_cell(0));  

                for(T_Cropped_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),relative_interval,p,true);iterator.Valid();iterator.Next()){
                    T_INDEX relative_index=iterator.Index()+T_INDEX(1);
					T weight=iterator.Weight();
                    TV weight_gradient=iterator.Weight_Gradient();
                    auto data=iterator.Current_Cell()._data;         
                    TV body_force=obj.gravity*p.mass*weight; 
                    if(p.eos) {
                        TV pressure_force;
                        if(affine){
                            T_INDEX cell_index=iterator.Current_Cell();
                            TV cell_X=grid.Center(cell_index);
                            TV diffX=cell_X-p.X0;
                            if(use_inv) pressure_force=weight*V0_P_FT*Dp_inv*diffX;
                            else pressure_force=weight*V0_P_FT*diffX*(T)4.*Nova_Utilities::Sqr(grid.one_over_dX(0));
                        }
                        else{
                            pressure_force=V0_P_FT*weight_gradient;
                        }
                        f0(data)+=pressure_force(0); f1(data)+=pressure_force(1); 
                    }
                    else { 
                        if(affine){
                            T_INDEX cell_index=iterator.Current_Cell();
                            TV cell_X=grid.Center(cell_index);
                            TV diffX=cell_X-p.X0;
                            TV inner_force;
                            if(use_inv) inner_force=-weight*V0_P_FT*Dp_inv*diffX;
                            else inner_force=-weight*V0_P_FT*Dp_inv*diffX;
                            f0(data)+=inner_force(0); f1(data)+=inner_force(1);
                        }
                        else{
                            TV inner_force=-V0_P_FT*weight_gradient;
                            f0(data)+=inner_force(0); f1(data)+=inner_force(1);
                        }
                    }
                    f0(data)+=body_force(0); f1(data)+=body_force(1);}}}}
    Explicit_Force_Helper<Struct_type,T,2>(hierarchy->Allocator(level),hierarchy->Blocks(level),f_channels,velocity_channels,velocity_star_channels,dt);
}
//######################################################################
// Apply_Explicit_Force
//######################################################################
template<class T> void MPM_Example<T,3>::
Apply_Explicit_Force(const int& oid)
{
    const Grid<T,3>& grid=hierarchy->Lattice(level);
    auto f0=hierarchy->Channel(level,f_channels(0));        
    auto f1=hierarchy->Channel(level,f_channels(1));
    auto f2=hierarchy->Channel(level,f_channels(2));
    Array<int> thread_cnt(threads);

    T_Object& obj=objects(oid);
#pragma omp parallel for
    for(int tid_process=0;tid_process<threads;++tid_process){
        const Interval<int>& thread_x_interval=x_intervals(tid_process);
        int& cnt=thread_cnt(tid_process);
        for(int tid_collect=0;tid_collect<threads;++tid_collect){
            const Array<int>& index=particle_bins(tid_process,tid_collect);
            for(int i=0;i<index.size();++i){
                T_Particle& p=particles(index(i)); T_INDEX& closest_cell=p.closest_cell;
                T eos_coefficient=(T)0.; T V0=p.volume; T_Mat P,F,V0_P_FT;
                const T_Mat& Dp_inv=p.Dp_inv;
                if(p.eos) {
                    const T_Mat& Fes=p.constitutive_model.Fes;
                    const T_Mat& Fes0=p.constitutive_model.Fes0;
                    const T_Mat& Fe=p.constitutive_model.Fe;
                    const T Je=Fe.Determinant(); 
                    const T Jes=Fes.Determinant();
                    const T Jes0=Fes0.Determinant();
                    eos_coefficient=V0/p.density*p.bulk_modulus*(pow(p.density,p.gamma)-(T)1);  // == sigma * V0 
                    if(std::abs(Jes)<1.e-5) V0_P_FT=T_Mat();
                    else V0_P_FT=Jes*eos_coefficient*(Fes.Inverse().Transposed());
                }
                else{ P=p.constitutive_model.P(),F=p.constitutive_model.Fes0; // sigma = 1/J * P * FT * V0 * J 
                    V0_P_FT=P.Times_Transpose(F)*V0;}
                const Interval<int> relative_interval=Interval<int>(thread_x_interval.min_corner-closest_cell(0),thread_x_interval.max_corner-closest_cell(0));  

                for(T_Cropped_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),relative_interval,p,true);iterator.Valid();iterator.Next()){
                    T_INDEX relative_index=iterator.Index()+T_INDEX(1);
					T weight=iterator.Weight();
                    TV weight_gradient=iterator.Weight_Gradient();
                    auto data=iterator.Current_Cell()._data;         
                    TV body_force=obj.gravity*p.mass*weight; 
                    if(p.eos) {
                        TV pressure_force;
                        if(affine){
                            T_INDEX cell_index=iterator.Current_Cell();
                            TV cell_X=grid.Center(cell_index);
                            TV diffX=cell_X-p.X0;
                            if(use_inv) pressure_force=weight*V0_P_FT*Dp_inv*diffX;
                            else pressure_force=weight*V0_P_FT*diffX*(T)4.*Nova_Utilities::Sqr(grid.one_over_dX(0));
                        }
                        else{
                            pressure_force=V0_P_FT*weight_gradient;
                        }
                        f0(data)+=pressure_force(0); 
                        f1(data)+=pressure_force(1); 
                        f2(data)+=pressure_force(2); 
                    }
                    else { 
                        if(affine){
                            T_INDEX cell_index=iterator.Current_Cell();
                            TV cell_X=grid.Center(cell_index);
                            TV diffX=cell_X-p.X0;
                            TV inner_force;
                            if(use_inv) inner_force=-weight*V0_P_FT*Dp_inv*diffX;
                            else inner_force=-weight*V0_P_FT*Dp_inv*diffX;
                            f0(data)+=inner_force(0); 
                            f1(data)+=inner_force(1);
                            f2(data)+=inner_force(2);
                        }
                        else{
                            TV inner_force=-V0_P_FT*weight_gradient;
                            f0(data)+=inner_force(0); 
                            f1(data)+=inner_force(1);
                            f2(data)+=inner_force(2);
                        }
                    }
                    f0(data)+=body_force(0); f1(data)+=body_force(1); f2(data)+=body_force(2);}}}}
    Explicit_Force_Helper<Struct_type,T,3>(hierarchy->Allocator(level),hierarchy->Blocks(level),f_channels,velocity_channels,velocity_star_channels,dt);
}
//######################################################################
// Populate_Simulated_Particles
//######################################################################
template<class T> void MPM_Example<T,2>::
Populate_Simulated_Particles(const int& oid)
{
    const T_Object& obj=objects(oid);
    simulated_particles.Clear();
    for(int i=0;i<obj.particle_ids.size();++i){
        const int pid=obj.particle_ids[i];
        if(particles(pid).valid)
            simulated_particles.Append(pid);
    }
}
//######################################################################
// Populate_Simulated_Particles
//######################################################################
template<class T> void MPM_Example<T,3>::
Populate_Simulated_Particles(const int& oid)
{
    const T_Object& obj=objects(oid);
    simulated_particles.Clear();
    for(int i=0;i<obj.particle_ids.size();++i){
        const int pid=obj.particle_ids[i];
        if(particles(pid).valid)
            simulated_particles.Append(pid);
    }
}
//######################################################################
// Max_Particle_Velocity
//######################################################################
template<class T> T MPM_Example<T,2>::
Max_Particle_Velocity() const
{
    Array<T> result_per_thread(threads);
#pragma omp parallel for
    for(unsigned i=0;i<particles.size();++i){
        const int tid=omp_get_thread_num();
        T& r=result_per_thread(tid); 
        r=std::max(r,particles(i).V.Norm_Squared());}
    T result=(T)0.;
    for(int tid=0;tid<threads;++tid) result=std::max(result,result_per_thread(tid));
    Log::cout<<"max v: "<<std::sqrt(result)<<std::endl;
    return std::sqrt(result);
}
//######################################################################
// Max_Particle_Velocity
//######################################################################
template<class T> T MPM_Example<T,3>::
Max_Particle_Velocity() const
{
    Array<T> result_per_thread(threads);
#pragma omp parallel for
    for(unsigned i=0;i<particles.size();++i){
        const int tid=omp_get_thread_num();
        T& r=result_per_thread(tid); 
        r=std::max(r,particles(i).V.Norm_Squared());}
    T result=(T)0.;
    for(int tid=0;tid<threads;++tid) result=std::max(result,result_per_thread(tid));
    Log::cout<<"max v: "<<std::sqrt(result)<<std::endl;
    return std::sqrt(result);
}
//######################################################################
// Limit_Dt
//######################################################################
template<class T> void MPM_Example<T,2>::
Limit_Dt(T& dt,const T time)
{
}
//######################################################################
// Limit_Dt
//######################################################################
template<class T> void MPM_Example<T,3>::
Limit_Dt(T& dt,const T time)
{
}
//######################################################################
// Update_Particle_Weights
//######################################################################
template<class T> void MPM_Example<T,2>::
Update_Particle_Weights(const int& oid)
{
    high_resolution_clock::time_point tb1=high_resolution_clock::now();
    const Grid<T,2>& grid=hierarchy->Lattice(level);
    T_Object& obj=objects(oid);
#pragma omp parallel for
        for(int i=0;i<obj.particle_ids.size();++i){
            const int pid=obj.particle_ids[i];
            T_Particle& p=particles(pid);
            if(p.valid){
                if(obj.first_time || obj.to_update)
                    p.Update_Weights(grid,true);
            }
        }
}
//######################################################################
// Update_Particle_Weights
//######################################################################
template<class T> void MPM_Example<T,3>::
Update_Particle_Weights(const int& oid)
{
    high_resolution_clock::time_point tb1=high_resolution_clock::now();
    const Grid<T,3>& grid=hierarchy->Lattice(level);
    T_Object& obj=objects(oid);
#pragma omp parallel for
        for(int i=0;i<obj.particle_ids.size();++i){
            const int pid=obj.particle_ids[i];
            T_Particle& p=particles(pid);
            if(obj.first_time || obj.to_update)
                p.Update_Weights(grid,true);
        }
    high_resolution_clock::time_point te1=high_resolution_clock::now();
	duration<double> dur1=duration_cast<duration<double>>(te1-tb1);
    update_rt+=dur1.count();
}
//######################################################################
// Group_Particles
//######################################################################
template<class T> void MPM_Example<T,2>::
Group_Particles(const int& oid)
{
    const T_Object& obj=objects(oid);
    const Grid<T,2>& grid=hierarchy->Lattice(level);
    T_INDEX min_corner=grid.Cell_Indices().min_corner, max_corner=grid.Cell_Indices().max_corner;
    x_intervals.resize(threads); x_intervals(0).min_corner=min_corner(0); x_intervals(threads-1).max_corner=max_corner(0);
    const T ratio=(max_corner(0)-min_corner(0)+1)/threads;
    for(int i=0;i<threads-1;++i){
        int n=min_corner(0)+(max_corner(0)-min_corner(0)+1)*(i+1)/threads;
        x_intervals(i).max_corner=n-1;
        x_intervals(i+1).min_corner=n;}
    for(int i=0;i<threads;++i) for(int j=0;j<threads;++j) particle_bins(i,j).Clear();
#pragma omp parallel for
    for(int i=0;i<obj.particle_ids.size();++i){
        const int pid=obj.particle_ids[i];
        if(particles(pid).valid){
            const T_INDEX& closest_cell=particles(pid).closest_cell;
            const int tid_collect=omp_get_thread_num();
            const Interval<int> particle_x_interval=Interval<int>(closest_cell(0)-1,closest_cell(0)+1);
            for(int tid_process=0;tid_process<threads;++tid_process){
                const Interval<int> thread_x_interval=x_intervals(tid_process);
                if(particle_x_interval.Intersection(thread_x_interval)) particle_bins(tid_process,tid_collect).Append(pid);
            }
        }
    }    
}
//######################################################################
// Group_Particles
//######################################################################
template<class T> void MPM_Example<T,3>::
Group_Particles(const int& oid)
{
    const T_Object& obj=objects(oid);
    const Grid<T,3>& grid=hierarchy->Lattice(level);
    T_INDEX min_corner=grid.Cell_Indices().min_corner, max_corner=grid.Cell_Indices().max_corner;
    x_intervals.resize(threads); x_intervals(0).min_corner=min_corner(0); x_intervals(threads-1).max_corner=max_corner(0);
    const T ratio=(max_corner(0)-min_corner(0)+1)/threads;
    for(int i=0;i<threads-1;++i){
        int n=min_corner(0)+(max_corner(0)-min_corner(0)+1)*(i+1)/threads;
        x_intervals(i).max_corner=n-1;
        x_intervals(i+1).min_corner=n;}
    for(int i=0;i<threads;++i) for(int j=0;j<threads;++j) particle_bins(i,j).Clear();
#pragma omp parallel for
    for(int i=0;i<obj.particle_ids.size();++i){
        const int pid=obj.particle_ids[i];
        if(particles(pid).valid){
            const T_INDEX& closest_cell=particles(pid).closest_cell;
            const int tid_collect=omp_get_thread_num();
            const Interval<int> particle_x_interval=Interval<int>(closest_cell(0)-1,closest_cell(0)+1);
            for(int tid_process=0;tid_process<threads;++tid_process){
                const Interval<int> thread_x_interval=x_intervals(tid_process);
                if(particle_x_interval.Intersection(thread_x_interval)) particle_bins(tid_process,tid_collect).Append(pid);
            }
        }
    }   
}
//######################################################################
// Update_Constitutive_Model_State
//######################################################################
template<class T> void MPM_Example<T,2>::
Update_Constitutive_Model_State(const int& oid)
{
    const T_Object& obj=objects(oid);
#pragma omp parallel for
    for(unsigned i=0;i<obj.particle_ids.size();++i){
        const int pid=obj.particle_ids[i]; 
        T_Particle &particle=particles(pid);    
        if(!particle.eos && particle.valid) particle.constitutive_model.Precompute(obj.to_update);}
}
//######################################################################
// Update_Constitutive_Model_State
//######################################################################
template<class T> void MPM_Example<T,3>::
Update_Constitutive_Model_State(const int& oid)
{
    const T_Object& obj=objects(oid);
#pragma omp parallel for
    for(unsigned i=0;i<obj.particle_ids.size();++i){
        const int pid=obj.particle_ids[i]; 
        T_Particle &particle=particles(pid);    
        if(!particle.eos) particle.constitutive_model.Precompute(obj.to_update);}
}
//######################################################################
// Compute_TL_Contact_Forces
//######################################################################
template<class T> void MPM_Example<T,2>::
Compute_TL_Contact_Forces(){}
//######################################################################
// Compute_TL_Contact_Forces
//######################################################################
template<class T> void MPM_Example<T,3>::
Compute_TL_Contact_Forces(){}
//######################################################################
// Grid_Based_Collision
//######################################################################
template<class T> void MPM_Example<T,2>::
Grid_Based_Collision(const int& oid)
{   
    T_Object& obj=objects(oid);
    TLMPM_Grid_Based_Collision_Helper<Struct_type,T,2>(hierarchy->Lattice(level),hierarchy->Allocator(level),hierarchy->Blocks(level),position_channels,velocity_star_channels,barriers,obj.sticky);
    if(sphere_boundaries.size()>0) Sphere_Collision_Helper<Struct_type,T,2>(hierarchy->Allocator(level),hierarchy->Blocks(level),position_channels,velocity_star_channels,sphere_boundaries);
    if(objects.size()>1) Barrier_Collision_Helper<Struct_type,T,2>(*hierarchy,hierarchy->Lattice(level),hierarchy->Allocator(level),hierarchy->Blocks(level),position_channels,barrier_velocity_channels,velocity_star_channels);
#if 1
    T omega=(T)0.; T vy=(T)0.;
    const Grid<T,2>& grid=hierarchy->Lattice(level);
    const T dx=grid.dX(0);
    T min_y=(T)1.e5; T max_y=(T)-1.e5;
    for(auto& p:particles){ const T location_y=p.X(1);
        if(location_y>max_y) max_y=location_y;
        else if(location_y<min_y) min_y=location_y;
    }
    Log::cout<<"Range: "<<min_y<<", "<<max_y<<std::endl;
    if(min_y<(T).1+(T)2.*dx) Fixed_Domain_Helper<Struct_type,T,2>(grid,hierarchy->Allocator(level),hierarchy->Blocks(level),position_channels,velocity_star_channels,fixed_domains,omega,vy);
#endif
}
//######################################################################
// Grid_Based_Collision
//######################################################################
template<class T> void MPM_Example<T,3>::
Grid_Based_Collision(const int& oid)
{   
    T_Object& obj=objects(oid);
    const Grid<T,3>& grid=hierarchy->Lattice(level);
    const T dx=grid.dX(0);
    TLMPM_Grid_Based_Collision_Helper<Struct_type,T,3>(hierarchy->Lattice(level),hierarchy->Allocator(level),hierarchy->Blocks(level),position_channels,velocity_star_channels,barriers,obj.sticky);
    if(cups.size()>0) Cup_Collision_Helper<Struct_type,T,3>(hierarchy->Allocator(level),hierarchy->Blocks(level),position_channels,velocity_star_channels,cups);
    if(sphere_boundaries.size()>0) Sphere_Collision_Helper<Struct_type,T,3>(hierarchy->Allocator(level),hierarchy->Blocks(level),position_channels,velocity_star_channels,sphere_boundaries);
    if(box_boundaries.size()>0) Box_Collision_Helper<Struct_type,T,3>(hierarchy->Allocator(level),hierarchy->Blocks(level),position_channels,velocity_star_channels,box_boundaries,sticky);
    // if(objects.size()>1) Barrier_Collision_Helper<Struct_type,T,3>(*hierarchy,hierarchy->Lattice(level),hierarchy->Allocator(level),hierarchy->Blocks(level),position_channels,barrier_velocity_channels,velocity_star_channels);
    // if(fixed_rings.size()>0) Fixed_Ring_Collision_Helper<Struct_type,T,3>(hierarchy->Lattice(level),hierarchy->Allocator(level),hierarchy->Blocks(level),position_channels,velocity_star_channels,fixed_rings);
    
#if 1
    T omega=(T)100.; T vy=(T)5.;
    if(fixed_domains.size()>0){
        T min_y=(T)1.e5; T max_y=(T)-1.e5;
        for(auto& p:particles){ const T location_y=p.X(1);
            if(location_y>max_y) max_y=location_y;
            else if(location_y<min_y) min_y=location_y;
        }
        Log::cout<<"Range: "<<min_y<<", "<<max_y<<std::endl;
        if(move_fixed_domains){
            if((max_y-min_y)<(T)1.e10 && max_y<domain.max_corner(1)*(T).75 && min_y>domain.max_corner(1)*(T).25){
            // if((max_y-min_y)<(T)1.e10){
            if(Base::test_number==26){
                // bottom
                fixed_domains(0).min_corner(1)=min_y;
                fixed_domains(0).max_corner(1)=min_y+(T)2.*dx;
                // top
                fixed_domains(1).min_corner(1)=max_y-(T)2.*dx;
                fixed_domains(1).max_corner(1)=max_y;
                Fixed_Domain_Helper<Struct_type,T,3>(grid,hierarchy->Allocator(level),hierarchy->Blocks(level),position_channels,velocity_star_channels,fixed_domains,omega,vy);
                }
            }
            else{
                T_Object& obj=objects(oid);
                obj.gravity=TV::Axis_Vector(0)*(T)-2.;
                fixed_domains(0).min_corner=TV(1); fixed_domains(0).max_corner=TV(-1);
                // fixed_domains(1).min_corner(1)-=dx; fixed_domains(0).max_corner(1)+=dx;
                omega=vy=(T)0.; move_fixed_domains=false;
                Fixed_Domain_Helper<Struct_type,T,3>(grid,hierarchy->Allocator(level),hierarchy->Blocks(level),position_channels,velocity_star_channels,fixed_domains,omega,vy);
            }
        }
        else{
            omega=vy=(T)0.;
            Fixed_Domain_Helper<Struct_type,T,3>(grid,hierarchy->Allocator(level),hierarchy->Blocks(level),position_channels,velocity_star_channels,fixed_domains,omega,vy);
        }
    }
#endif
}
//######################################################################
// Compute_Energy
//######################################################################
template<class T> void MPM_Example<T,2>::
Compute_Energy(T time)
{
    Array<T> thread_KE(threads, (T)0);
    Array<T> thread_PE(threads, (T)0);

#pragma omp parallel for
    for(unsigned i=0;i<simulated_particles.size();++i){
        const int id=simulated_particles(i); 
        T_Particle &p=particles(id);
        int current_thread = omp_get_thread_num();
        if(p.eos){}
        else{
	        p.KE = p.Compute_Kinetic_Energy();
	        p.PE = p.Compute_Potential_Energy(gravity.Norm(), E);

            thread_KE(current_thread) += p.KE;
            thread_PE(current_thread) += p.PE;
        }
    }

    T total_Kinetic_Energy = (T)0.;
    T total_Potential_Energy = (T)0.;
    T total_Energy = (T)0.;
    for(int i=0; i<threads; ++i){
        total_Kinetic_Energy+=thread_KE(i);
        total_Potential_Energy+=thread_PE(i);
    }
    total_Energy = total_Kinetic_Energy + total_Potential_Energy;

#if 0
    //output the energy
    std::string file=output_directory+"/energy_test"+std::to_string(Base::test_number)+".txt";
    std::FILE *fp = std::fopen(file.c_str(),"a");
    fprintf(fp,"%f %f %f %f\n", time, total_Kinetic_Energy, total_Potential_Energy, total_Energy);
    std::fclose(fp);
#endif
}
//######################################################################
// Compute_Energy
//######################################################################
template<class T> void MPM_Example<T,3>::
Compute_Energy(T time)
{
    Array<T> thread_KE(threads, (T)0);
    Array<T> thread_PE(threads, (T)0);

#pragma omp parallel for
    for(unsigned i=0;i<simulated_particles.size();++i){
        const int id=simulated_particles(i); 
        T_Particle &p=particles(id);
        int current_thread = omp_get_thread_num();
        if(p.eos){}
        else{
	        p.KE = p.Compute_Kinetic_Energy();
	        p.PE = p.Compute_Potential_Energy(gravity.Norm(), E);

            thread_KE(current_thread) += p.KE;
            thread_PE(current_thread) += p.PE;
        }
    }

    T total_Kinetic_Energy = (T)0.;
    T total_Potential_Energy = (T)0.;
    T total_Energy = (T)0.;
    for(int i=0; i<threads; ++i){
        total_Kinetic_Energy+=thread_KE(i);
        total_Potential_Energy+=thread_PE(i);
    }
    total_Energy = total_Kinetic_Energy + total_Potential_Energy;

#if 0
    //output the energy
    std::string file=output_directory+"/energy_test"+std::to_string(Base::test_number)+"_rs"+std::to_string(rs)+"_rmin"+std::to_string(rmin)+"_rmax"+std::to_string(rmax)+".txt";
    std::FILE *fp = std::fopen(file.c_str(),"a");
    fprintf(fp,"%f %f %f %f\n", time, total_Kinetic_Energy, total_Potential_Energy, total_Energy);
    std::fclose(fp);
#endif
}
//######################################################################
// Compute_von_Mises_Stress
//######################################################################
template<class T> void MPM_Example<T,2>::
Compute_von_Mises_Stress()
{
#pragma omp parallel for
    for(unsigned i=0;i<simulated_particles.size();++i){
        const int id=simulated_particles(i); 
        T_Particle &p=particles(id);
        p.von_Mises = p.von_Mises_Stress_2D();
    }
}
//######################################################################
// Compute_von_Mises_Stress
//######################################################################
template<class T> void MPM_Example<T,3>::
Compute_von_Mises_Stress()
{
#pragma omp parallel for
    for(unsigned i=0;i<simulated_particles.size();++i){
        const int id=simulated_particles(i); 
        T_Particle &p=particles(id);
        p.von_Mises = p.von_Mises_Stress_3D();
    }
}
//######################################################################
// Print_von_Mises_Stress
//######################################################################
template<class T> void MPM_Example<T,2>::
Print_von_Mises_Stress(const int frame)
{
    std::string file=output_directory+"/von_mises/von_mises_stress"+std::to_string(frame)+".txt";
    std::FILE *fp = std::fopen(file.c_str(),"a");
    for(unsigned i=0;i<simulated_particles.size();++i){
        const int id=simulated_particles(i);
        T_Particle &p=particles(id);
        fprintf(fp,"%f %f %f\n", p.X(0), p.X(1), p.von_Mises);
    }
    std::fclose(fp);
}
//######################################################################
// Print_von_Mises_Stress
//######################################################################
template<class T> void MPM_Example<T,3>::
Print_von_Mises_Stress(const int frame)
{
    std::string file=output_directory+"/von_mises/von_mises_stress"+std::to_string(frame)+".txt";
    std::FILE *fp = std::fopen(file.c_str(),"a");
    for(unsigned i=0;i<simulated_particles.size();++i){
        const int id=simulated_particles(i);
        T_Particle &p=particles(id);
        fprintf(fp,"%f %f %f %f\n", p.X(0), p.X(1), p.X(2), p.von_Mises);
    }
    std::fclose(fp);
}
//######################################################################
// Add_Object
//######################################################################
template<class T> void MPM_Example<T,3>::
Add_Object()
{
    if(add_bunny){
        if(Base::current_frame>1 && Base::current_frame%30==0 && Base::current_frame<130){
            int oid=objects.size();
            int p_cnt=particles.size();
            T_Object obj;
            obj.gravity=TV::Axis_Vector(1)*(T)-9.8;
            int current_size=particles.size()-vis_size;
            TV random_perturbation=random.Get_Uniform_Vector(Box<T,3>(TV({-.1,0.,-.1}),TV({.1,0.,.1})));
            if(current_size<(1+Base::current_frame/30)*particles_bak.size()){
                for(auto p: particles_bak) {
                    Log::cout<<"Adding"<<std::endl;
                    p.valid=true;
                    TV X=p.X;
                    Vector<T,2> X_zx=Vector<T,2>({X(2)-(T).5,X(0)-(T).5});
                    const T angle=pi/3.*(Base::current_frame/30);
                    Matrix<T,2> R(cos(angle),-sin(angle),sin(angle),cos(angle));
                    X_zx=R*X_zx;
                    p.X=TV({(T).5+X_zx(1),X(1),(T).5+X_zx(0)})+random_perturbation;
                    p.X0=p.X;
                    p.V(1)=(T)-.1;
                    p.oid=oid;
                    p.eos=false;
                    p.constitutive_model.Compute_Lame_Parameters(E,nu);
                    p.constitutive_model.plastic=true;
                    p.constitutive_model.stretching_yield=(T)1.+stretching;
                    p.constitutive_model.compression_yield=(T)1.-compression;
                    p.constitutive_model.hardening_factor=hardening_factor;
                    particles.Append(p);
                    obj.particle_ids.Append(p_cnt++);
                }
                objects.Append(obj);
            }
        }
    }
}
//######################################################################
// Register_Options
//######################################################################
template<class T> void MPM_Example<T,2>::
Register_Options()
{
    Base::Register_Options();
    parse_args->Add_Option_Argument("-ui","use Dp_inv");
    parse_args->Add_Option_Argument("-test","test");
    parse_args->Add_Option_Argument("-im","implicit");
    parse_args->Add_Option_Argument("-affine","Conserve angular momentum with affine approach");
    parse_args->Add_Option_Argument("-tilde","use tilde");
    parse_args->Add_Option_Argument("-vonMises","Print Von Mises stress for every frame");
    parse_args->Add_Integer_Argument("-npc",64,"Number of particles per cell");
    parse_args->Add_Integer_Argument("-threads",1,"Number of threads for OpenMP to use");
    parse_args->Add_Integer_Argument("-levels",1,"Number of levels in the SPGrid hierarchy.");
    parse_args->Add_Double_Argument("-cfl",(T)0.1,"CFL number.");
    parse_args->Add_Double_Argument("-dt",(T)0.1,"CFL number.");
    parse_args->Add_Double_Argument("-flip",(T)0.95,"flip");
    parse_args->Add_Double_Argument("-E",(T)1000,"E.");
    parse_args->Add_Double_Argument("-nu",(T).2,"nu.");
    parse_args->Add_Vector_2D_Argument("-size",Vector<double,2>(32.),"n","Grid resolution");
 
    parse_args->Add_Double_Argument("-pct",(T).2,"percentage");
}
//######################################################################
// Register_Options
//######################################################################
template<class T> void MPM_Example<T,3>::
Register_Options()
{
    Base::Register_Options();
    parse_args->Add_Option_Argument("-ui","use Dp_inv");
    parse_args->Add_Option_Argument("-test","test");
    parse_args->Add_Option_Argument("-im","implicit");
    parse_args->Add_Option_Argument("-affine","Conserve angular momentum with affine approach");
    parse_args->Add_Option_Argument("-tilde","use tilde");
    parse_args->Add_Option_Argument("-vonMises","Print Von Mises stress for every frame");
    parse_args->Add_Integer_Argument("-npc",64,"Number of particles per cell");
    parse_args->Add_Integer_Argument("-threads",1,"Number of threads for OpenMP to use");
    parse_args->Add_Integer_Argument("-levels",1,"Number of levels in the SPGrid hierarchy.");
    parse_args->Add_Double_Argument("-cfl",(T)0.1,"CFL number.");
    parse_args->Add_Double_Argument("-dt",(T)0.1,"CFL number.");
    parse_args->Add_Double_Argument("-flip",(T)0.95,"flip");
    parse_args->Add_Double_Argument("-E",(T)14000,"E.");
    parse_args->Add_Double_Argument("-nu",(T).2,"nu.");
    parse_args->Add_Double_Argument("-h",(T)10.,"hardening");
    parse_args->Add_Double_Argument("-comp",(T)2.5e-2,"compression factor.");
    parse_args->Add_Option_Argument("-sticky","sticky.");
    parse_args->Add_Double_Argument("-stretch",(T)7.5e-3,"stretching factor.");

    parse_args->Add_Vector_3D_Argument("-size",Vector<double,3>(32.),"n","Grid resolution");
    parse_args->Add_Double_Argument("-pct",(T).2,"percentage");
}
//######################################################################
// Test
//######################################################################
template<class T> void MPM_Example<T,2>::
Test(){}
//######################################################################
// Test
//######################################################################
template<class T> void MPM_Example<T,3>::
Test(){}
//######################################################################
// Parse_Options
//######################################################################
template<class T> void MPM_Example<T,2>::
Parse_Options()
{
    Base::Parse_Options();
    threads=parse_args->Get_Integer_Value("-threads");
    npc=parse_args->Get_Integer_Value("-npc");
    omp_set_num_threads(threads);
    Base::test_number=parse_args->Get_Integer_Value("-test_number");
    affine=parse_args->Get_Option_Value("-affine");
    use_inv=parse_args->Get_Option_Value("-ui");
    vonMises=parse_args->Get_Option_Value("-vonMises");
    cfl=parse_args->Get_Double_Value("-cfl");
    const_dt=parse_args->Get_Double_Value("-dt");
    flip=parse_args->Get_Double_Value("-flip");
    E=parse_args->Get_Double_Value("-E");
    nu=parse_args->Get_Double_Value("-nu");
    const_dt=parse_args->Get_Double_Value("-dt");
    levels=parse_args->Get_Integer_Value("-levels");
    auto cell_counts_2d=parse_args->Get_Vector_2D_Value("-size"); for(int v=0;v<2;++v) counts(v)=cell_counts_2d(v);
    percentage=parse_args->Get_Double_Value("-pct");
    implicit_solve=parse_args->Get_Option_Value("-im");
}
//######################################################################
// Parse_Options
//######################################################################
template<class T> void MPM_Example<T,3>::
Parse_Options()
{
    Base::Parse_Options();
    threads=parse_args->Get_Integer_Value("-threads");
    npc=parse_args->Get_Integer_Value("-npc");
    omp_set_num_threads(threads);
    Base::test_number=parse_args->Get_Integer_Value("-test_number");
    affine=parse_args->Get_Option_Value("-affine");
    use_inv=parse_args->Get_Option_Value("-ui");
    vonMises=parse_args->Get_Option_Value("-vonMises");
    cfl=parse_args->Get_Double_Value("-cfl");
    flip=parse_args->Get_Double_Value("-flip");
    
    E=parse_args->Get_Double_Value("-E");
    nu=parse_args->Get_Double_Value("-nu");
    hardening_factor=parse_args->Get_Double_Value("-h");
    stretching=parse_args->Get_Double_Value("-stretch");
    compression=parse_args->Get_Double_Value("-comp");
    sticky=parse_args->Get_Option_Value("-sticky");

    const_dt=parse_args->Get_Double_Value("-dt");
    levels=parse_args->Get_Integer_Value("-levels");
    auto cell_counts_3d=parse_args->Get_Vector_3D_Value("-size"); for(int v=0;v<3;++v) counts(v)=cell_counts_3d(v);
    percentage=parse_args->Get_Double_Value("-pct");
    implicit_solve=parse_args->Get_Option_Value("-im");
}
//######################################################################
// Allocate_Particle
//######################################################################
template<class T> int MPM_Example<T,2>::
Allocate_Particle(bool add_to_simulation)
{
    int id=0;
    if(invalid_particles.size()){
        invalid_particles.Pop_Back();
        id=invalid_particles.size();
        particles(id).Initialize();}
    else {T_Particle p;
        id=particles.size();
        particles.Append(p);}
    if(add_to_simulation){
        simulated_particles.Append(id);
        waiting_particles.Append(id);}
    else particles(id).valid=false;
    return id;
}
//######################################################################
// Allocate_Particle
//######################################################################
template<class T> int MPM_Example<T,3>::
Allocate_Particle(bool add_to_simulation)
{
    int id=0;
    if(invalid_particles.size()){
        invalid_particles.Pop_Back();
        id=invalid_particles.size();
        particles(id).Initialize();}
    else {T_Particle p;
        id=particles.size();
        particles.Append(p);}
    if(add_to_simulation){
        simulated_particles.Append(id);
        waiting_particles.Append(id);}
    else particles(id).valid=false;
    return id;
}
//######################################################################
// Write_Output_Files
//######################################################################
template<class T> void MPM_Example<T,2>::
Write_Output_Files(const int frame) const
{
    if(frame==first_frame){std::string deformables_filename=output_directory+"/common/metadata.mpm2d";
        File_Utilities::Write_To_Text_File(deformables_filename,std::to_string(frame));}

    File_Utilities::Create_Directory(output_directory+"/"+std::to_string(frame));
    File_Utilities::Write_To_Text_File(output_directory+"/"+std::to_string(frame)+"/frame_title",frame_title);

    File_Utilities::Write_To_File(output_directory+"/"+std::to_string(frame)+"/particles",particles);

    // write hierarchy
    File_Utilities::Write_To_Text_File(output_directory+"/"+std::to_string(frame)+"/levels",levels);
    hierarchy->Write_Hierarchy(output_directory,frame);
}
//######################################################################
// Write_Output_Files
//######################################################################
template<class T> void MPM_Example<T,3>::
Write_Output_Files(const int frame) const
{
    if(frame==first_frame){std::string deformables_filename=output_directory+"/common/metadata.mpm3d";
        File_Utilities::Write_To_Text_File(deformables_filename,std::to_string(frame));}

    File_Utilities::Create_Directory(output_directory+"/"+std::to_string(frame));
    File_Utilities::Write_To_Text_File(output_directory+"/"+std::to_string(frame)+"/frame_title",frame_title);

    File_Utilities::Write_To_File(output_directory+"/"+std::to_string(frame)+"/particles",particles);
    File_Utilities::Write_To_File(output_directory+"/"+std::to_string(frame)+"/boat_particles",boat.obj_particles);

    // write hierarchy
    File_Utilities::Write_To_Text_File(output_directory+"/"+std::to_string(frame)+"/levels",levels);
    hierarchy->Write_Hierarchy(output_directory,frame);
}
//######################################################################
// Read_Output_Files
//######################################################################
template<class T> void MPM_Example<T,2>::
Read_Output_Files(const int& frame)
{
    File_Utilities::Read_From_File(output_directory+"/"+std::to_string(frame)+"/particles",particles);
}
//######################################################################
// Read_Output_Files
//######################################################################
template<class T> void MPM_Example<T,3>::
Read_Output_Files(const int& frame)
{
    File_Utilities::Read_From_File(output_directory+"/"+std::to_string(frame)+"/particles",particles);
}
//######################################################################
// Apply_Contact_Force
//######################################################################
template<class T> void MPM_Example<T,2>::
Apply_Contact_Force()
{
    ALL_Populate_Simulated_Particles();
    ALL_Initialize_SPGrid();
    ALL_particle_bins.Resize(threads,threads);
    ALL_Update_Particle_Weights();
    ALL_Group_Particles();
    ALL_Rasterize_Voxels();
    ALL_Rasterize();
}
//######################################################################
// Apply_Contact_Force
//######################################################################
template<class T> void MPM_Example<T,3>::
Apply_Contact_Force()
{
    high_resolution_clock::time_point tb=high_resolution_clock::now();
    for(auto& obj:objects) obj.Update_Bounding_Box(threads,counts,particles);
    bool contacted=false;
    for(int i=0;i<objects.size();++i){
        const T_Object& obj1=objects(i);
        const Range<T,3>& bounding_box1=obj1.bounding_box;
        for(int j=i+1;j<objects.size();++j){
            const T_Object& obj2=objects(j);
            const Range<T,3>& bounding_box2=obj2.bounding_box;
            if(bounding_box1.Intersection(bounding_box2)){
                contacted=true;
                break;
            }
        }
    }
    if(contacted){
        ALL_Populate_Simulated_Particles();
        ALL_Initialize_SPGrid();
        ALL_particle_bins.Resize(threads,threads);
        ALL_Update_Particle_Weights();
        ALL_Group_Particles();
        ALL_Rasterize_Voxels();
        ALL_Rasterize();
    }
    
    high_resolution_clock::time_point te=high_resolution_clock::now();
	duration<double> dur=duration_cast<duration<double>>(te-tb);
    contact_rt+=dur.count();
    contact_cnt++;
}
//######################################################################
// ALL_Populate_Simulated_Particles
//######################################################################
template<class T> void MPM_Example<T,2>::
ALL_Populate_Simulated_Particles()
{
    ALL_simulated_particles.Clear();
    for(int oid=0;oid<objects.size();++oid){
        const T_Object& obj=objects(oid);
        for(int i=0;i<obj.particle_ids.size();++i){
            const int pid=obj.particle_ids[i];
            if(particles(pid).valid)
                ALL_simulated_particles.Append(pid);
        }
    }
}
//######################################################################
// ALL_Populate_Simulated_Particles
//######################################################################
template<class T> void MPM_Example<T,3>::
ALL_Populate_Simulated_Particles()
{
    ALL_simulated_particles.Clear();
    for(int oid=0;oid<objects.size();++oid){
        const T_Object& obj=objects(oid);
        for(int i=0;i<obj.particle_ids.size();++i){
            const int pid=obj.particle_ids[i];
            if(particles(pid).valid)
                ALL_simulated_particles.Append(pid);
        }
    }
}
//######################################################################
// ALL_Populate_Simulated_Particles
//######################################################################
template<class T> void MPM_Example<T,2>::
ALL_Initialize_SPGrid()
{
    ALL_Compute_Bounding_Box(ALL_bbox);
    if(ALL_hierarchy!=nullptr) delete ALL_hierarchy;
    ALL_hierarchy=new Hierarchy(counts,domain,levels);
}
//######################################################################
// ALL_Populate_Simulated_Particles
//######################################################################
template<class T> void MPM_Example<T,3>::
ALL_Initialize_SPGrid()
{
    ALL_Compute_Bounding_Box(ALL_bbox);
    if(ALL_hierarchy!=nullptr) delete ALL_hierarchy;
    ALL_hierarchy=new Hierarchy(counts,domain,levels);
}
//######################################################################
// ALL_Compute_Bounding_Box
//######################################################################
template<class T> void MPM_Example<T,2>::
ALL_Compute_Bounding_Box(Range<T,2>& ALL_bbox)
{
    Array<TV> min_corner_per_thread(threads,domain.max_corner);
    Array<TV> max_corner_per_thread(threads,domain.min_corner);
    
#pragma omp parallel for
    for(unsigned i=0;i<ALL_simulated_particles.size();++i){const int tid=omp_get_thread_num();
        const int pid=ALL_simulated_particles(i);
        const T_Particle& p=particles(pid);
        TV& current_min_corner=min_corner_per_thread(tid);
        TV& current_max_corner=max_corner_per_thread(tid);
        const TV& p_X=p.X;
        for(int v=0;v<2;++v){
            const T dd=(T)2./counts(v);
            current_min_corner(v)=std::min(current_min_corner(v),p_X(v)-dd);
            current_max_corner(v)=std::max(current_max_corner(v),p_X(v)+dd);}
        const TV& p_X0=p.X0;
        for(int v=0;v<2;++v){
            const T dd=(T)2./counts(v);
            current_min_corner(v)=std::min(current_min_corner(v),p_X0(v)-dd);
            current_max_corner(v)=std::max(current_max_corner(v),p_X0(v)+dd);}
    }

    for(int v=0;v<2;++v){
        ALL_bbox.min_corner(v)=min_corner_per_thread(0)(v);
        ALL_bbox.max_corner(v)=max_corner_per_thread(0)(v);
    }

    for(int tid=1;tid<threads;++tid) for(int v=0;v<2;++v){
        ALL_bbox.min_corner(v)=std::min(ALL_bbox.min_corner(v),min_corner_per_thread(tid)(v));
        ALL_bbox.max_corner(v)=std::max(ALL_bbox.max_corner(v),max_corner_per_thread(tid)(v));}
    
    for(int v=0;v<2;++v){ 
        ALL_bbox.min_corner(v)=std::max(domain.min_corner(v),ALL_bbox.min_corner(v));
        ALL_bbox.max_corner(v)=std::min(domain.max_corner(v),ALL_bbox.max_corner(v));}
}
//######################################################################
// ALL_Compute_Bounding_Box
//######################################################################
template<class T> void MPM_Example<T,3>::
ALL_Compute_Bounding_Box(Range<T,3>& ALL_bbox)
{
    Array<TV> min_corner_per_thread(threads,domain.max_corner);
    Array<TV> max_corner_per_thread(threads,domain.min_corner);
    
#pragma omp parallel for
    for(unsigned i=0;i<ALL_simulated_particles.size();++i){const int tid=omp_get_thread_num();
        const int pid=ALL_simulated_particles(i);
        const T_Particle& p=particles(pid);
        TV& current_min_corner=min_corner_per_thread(tid);
        TV& current_max_corner=max_corner_per_thread(tid);
        const TV& p_X=p.X;
        for(int v=0;v<3;++v){
            const T dd=(T)2./counts(v);
            current_min_corner(v)=std::min(current_min_corner(v),p_X(v)-dd);
            current_max_corner(v)=std::max(current_max_corner(v),p_X(v)+dd);}}

    for(int v=0;v<3;++v){
        ALL_bbox.min_corner(v)=min_corner_per_thread(0)(v);
        ALL_bbox.max_corner(v)=max_corner_per_thread(0)(v);
    }

    for(int tid=1;tid<threads;++tid) for(int v=0;v<3;++v){
        ALL_bbox.min_corner(v)=std::min(ALL_bbox.min_corner(v),min_corner_per_thread(tid)(v));
        ALL_bbox.max_corner(v)=std::max(ALL_bbox.max_corner(v),max_corner_per_thread(tid)(v));}
    
    for(int v=0;v<3;++v){ 
        ALL_bbox.min_corner(v)=std::max(domain.min_corner(v),ALL_bbox.min_corner(v));
        ALL_bbox.max_corner(v)=std::min(domain.max_corner(v),ALL_bbox.max_corner(v));}
}
//######################################################################
// ALL_Update_Particle_Weights
//######################################################################
template<class T> void MPM_Example<T,2>::
ALL_Update_Particle_Weights()
{
    const Grid<T,2>& grid=ALL_hierarchy->Lattice(level);
#pragma omp parallel for
    for(int i=0;i<ALL_simulated_particles.size();++i){
        const int pid=ALL_simulated_particles(i);
        T_Particle& p=particles(pid);
        p.ALL_Update_Closest_Cell(grid);
    }
}
//######################################################################
// ALL_Update_Particle_Weights
//######################################################################
template<class T> void MPM_Example<T,3>::
ALL_Update_Particle_Weights()
{
    const Grid<T,3>& grid=ALL_hierarchy->Lattice(level);
#pragma omp parallel for
    for(int i=0;i<ALL_simulated_particles.size();++i){
        const int pid=ALL_simulated_particles(i);
        T_Particle& p=particles(pid);
        p.ALL_Update_Closest_Cell(grid);
    }
}
//######################################################################
// ALL_Group_Particles
//######################################################################
template<class T> void MPM_Example<T,2>::
ALL_Group_Particles()
{
    const Grid<T,2>& grid=ALL_hierarchy->Lattice(level);
    T_INDEX min_corner=grid.Cell_Indices().min_corner, max_corner=grid.Cell_Indices().max_corner;
    ALL_x_intervals.resize(threads); 
    ALL_x_intervals(0).min_corner=min_corner(0); 
    ALL_x_intervals(threads-1).max_corner=max_corner(0);
    const T ratio=(max_corner(0)-min_corner(0)+1)/threads;
    for(int i=0;i<threads-1;++i){
        int n=min_corner(0)+(max_corner(0)-min_corner(0)+1)*(i+1)/threads;
        ALL_x_intervals(i).max_corner=n-1;
        ALL_x_intervals(i+1).min_corner=n;}
    for(int i=0;i<threads;++i) for(int j=0;j<threads;++j) ALL_particle_bins(i,j).Clear();
#pragma omp parallel for
    for(int i=0;i<ALL_simulated_particles.size();++i){
        const int pid=ALL_simulated_particles(i);
        const T_INDEX& closest_cell=particles(pid).closest_cell_X;
        const int tid_collect=omp_get_thread_num();
        const Interval<int> particle_x_interval=Interval<int>(closest_cell(0)-1,closest_cell(0)+1);
        for(int tid_process=0;tid_process<threads;++tid_process){
            const Interval<int> thread_x_interval=ALL_x_intervals(tid_process);
            if(particle_x_interval.Intersection(thread_x_interval)) ALL_particle_bins(tid_process,tid_collect).Append(pid);}}    
}
//######################################################################
// ALL_Group_Particles
//######################################################################
template<class T> void MPM_Example<T,3>::
ALL_Group_Particles()
{
    const Grid<T,3>& grid=ALL_hierarchy->Lattice(level);
    T_INDEX min_corner=grid.Cell_Indices().min_corner, max_corner=grid.Cell_Indices().max_corner;
    ALL_x_intervals.resize(threads); 
    ALL_x_intervals(0).min_corner=min_corner(0); 
    ALL_x_intervals(threads-1).max_corner=max_corner(0);
    const T ratio=(max_corner(0)-min_corner(0)+1)/threads;
    for(int i=0;i<threads-1;++i){
        int n=min_corner(0)+(max_corner(0)-min_corner(0)+1)*(i+1)/threads;
        ALL_x_intervals(i).max_corner=n-1;
        ALL_x_intervals(i+1).min_corner=n;}
    for(int i=0;i<threads;++i) for(int j=0;j<threads;++j) ALL_particle_bins(i,j).Clear();
#pragma omp parallel for
    for(int i=0;i<ALL_simulated_particles.size();++i){
        const int pid=ALL_simulated_particles(i);
        const T_INDEX& closest_cell=particles(pid).closest_cell_X;
        const int tid_collect=omp_get_thread_num();
        const Interval<int> particle_x_interval=Interval<int>(closest_cell(0)-1,closest_cell(0)+1);
        for(int tid_process=0;tid_process<threads;++tid_process){
            const Interval<int> thread_x_interval=ALL_x_intervals(tid_process);
            if(particle_x_interval.Intersection(thread_x_interval)) ALL_particle_bins(tid_process,tid_collect).Append(pid);}}    
}
//######################################################################
// ALL_Rasterize_Voxels
//######################################################################
template<class T> void MPM_Example<T,2>::
ALL_Rasterize_Voxels()
{
    using Cell_Iterator             = Grid_Iterator_Cell<T,2>;
    using Hierarchy_Initializer     = Grid_Hierarchy_Initializer<Struct_type,T,2>;
    const Grid<T,2>& grid=ALL_hierarchy->Lattice(0);
    Range<int,2> bounding_grid_cells(grid.Clamp_To_Cell(ALL_bbox.min_corner),grid.Clamp_To_Cell(ALL_bbox.max_corner));
    for(Cell_Iterator iterator(grid,bounding_grid_cells);iterator.Valid();iterator.Next())
        ALL_hierarchy->Activate_Cell(0,iterator.Cell_Index(),Cell_Type_Dirichlet);
    ALL_hierarchy->Update_Block_Offsets();
    ALL_hierarchy->Initialize_Red_Black_Partition(2*threads);
}
//######################################################################
// ALL_Rasterize_Voxels
//######################################################################
template<class T> void MPM_Example<T,3>::
ALL_Rasterize_Voxels()
{
    using Cell_Iterator             = Grid_Iterator_Cell<T,3>;
    using Hierarchy_Initializer     = Grid_Hierarchy_Initializer<Struct_type,T,2>;
    const Grid<T,3>& grid=ALL_hierarchy->Lattice(0);
    Range<int,3> bounding_grid_cells(grid.Clamp_To_Cell(ALL_bbox.min_corner),grid.Clamp_To_Cell(ALL_bbox.max_corner));
    for(Cell_Iterator iterator(grid,bounding_grid_cells);iterator.Valid();iterator.Next())
        ALL_hierarchy->Activate_Cell(0,iterator.Cell_Index(),Cell_Type_Dirichlet);
    ALL_hierarchy->Update_Block_Offsets();
    ALL_hierarchy->Initialize_Red_Black_Partition(2*threads);
}
//######################################################################
// ALL_Rasterize_Voxels
//######################################################################
template<class T> void MPM_Example<T,2>::
ALL_Rasterize()
{
    const int number_of_objects=objects.size();
    Array<Array<Array<T> > > Object_volume=Array<Array<Array<T> > >(number_of_objects,Array<Array<T> >(counts(0),Array<T>(counts(1),(T)0.)));
    Array<Array<T> > ALL_volume=Array<Array<T> >(counts(0),Array<T>(counts(1),(T)0.));

    Array<Array<Array<T> > > Object_mass=Array<Array<Array<T> > >(number_of_objects,Array<Array<T> >(counts(0),Array<T>(counts(1),(T)0.)));
    Array<Array<T> > ALL_mass=Array<Array<T> >(counts(0),Array<T>(counts(1),(T)0.));

    Array<Array<Array<TV> > > Object_p=Array<Array<Array<TV> > >(number_of_objects,Array<Array<TV> >(counts(0),Array<TV>(counts(1),TV())));
    Array<Array<TV> > ALL_p=Array<Array<TV> >(counts(0),Array<TV>(counts(1),TV()));

    Array<Array<Array<T> > > marker=Array<Array<Array<T> > >(number_of_objects,Array<Array<T> >(counts(0),Array<T>(counts(1),(T)0.)));
    const Grid<T,2> grid(counts,domain);
#pragma omp parallel for
    for(int tid_process=0;tid_process<threads;++tid_process){
        const Interval<int> thread_x_interval=ALL_x_intervals(tid_process);
        for(int tid_collect=0;tid_collect<threads;++tid_collect){
            Array<int>& index=ALL_particle_bins(tid_process,tid_collect);
            for(int i=0;i<index.size();++i){
                T_Particle& p=particles(index(i));T_INDEX& closest_cell=p.closest_cell_X;
                const Interval<int> relative_interval=Interval<int>(thread_x_interval.min_corner-closest_cell(0),thread_x_interval.max_corner-closest_cell(0));
                for(T_Cropped_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),relative_interval,p,false);iterator.Valid();iterator.Next()){
                    const T& p_volume=p.volume;
                    const T p_mass=p.mass; const TV p_V=p.V;
                    const T weight=iterator.Weight(); 

                    T_INDEX cell_index=iterator.Current_Cell()-T_INDEX(1);
                    int oid=p.oid;
                    Object_mass(oid)(cell_index(0))(cell_index(1))+=weight*p_mass;
                    ALL_mass(cell_index(0))(cell_index(1))+=weight*p_mass;

                    Object_volume(oid)(cell_index(0))(cell_index(1))+=weight*p_volume;
                    ALL_volume(cell_index(0))(cell_index(1))+=weight*p_volume;

                    Object_p(oid)(cell_index(0))(cell_index(1))+=weight*p_mass*p_V;
                    ALL_p(cell_index(0))(cell_index(1))+=weight*p_mass*p_V;
                }
            }
        }
    }


#pragma omp parallel for
    for(int oid=0;oid<number_of_objects;++oid){
        T_Object& obj=objects(oid);
        obj.barriers.Clear();
        obj.velocities.Clear();
        for(int i=0;i<obj.particle_ids.size();++i){
            const int pid=obj.particle_ids[i];
            T_Particle& p=particles(pid);
            p.ALL_Update_Closest_Cell(grid);
            const T_INDEX& closest_cell=p.closest_cell_X;
            for(Range_Iterator<2> iterator(T_INDEX(-1),T_INDEX(1));iterator.Valid();iterator.Next()){
                T_INDEX cur_cell=closest_cell+iterator.Index()-T_INDEX(1);      // 0-index
                const T& cell_Object_volume=Object_volume(oid)(cur_cell(0))(cur_cell(1));
                const T& cell_ALL_volume=ALL_volume(cur_cell(0))(cur_cell(1));
                const T cell_Other_volume=cell_ALL_volume-cell_Object_volume;
                const T flag=marker(oid)(cur_cell(0))(cur_cell(1));
                if(flag<(T).5 && cell_Other_volume>(T)0. && cell_Object_volume>(T)0.){
                    T_INDEX index=cur_cell+T_INDEX(1);      // 1-indexed
                    // Log::cout<<index<<std::endl;
                    const T& obj_mass=Object_mass(oid)(cur_cell(0))(cur_cell(1));
                    const TV& obj_p=Object_p(oid)(cur_cell(0))(cur_cell(1));
                    const TV obj_V=obj_p/obj_mass;

                    const T& all_mass=ALL_mass(cur_cell(0))(cur_cell(1));
                    const TV& all_p=ALL_p(cur_cell(0))(cur_cell(1));
                    
                    const T other_mass=all_mass-obj_mass;
                    const TV other_p=all_p-obj_p;
                    const TV other_V=other_p/other_mass;
                
                    TV normal=TV();
                    for(int axis=0;axis<2;++axis){
                        T axis_normal=(T)0.;
                        for(int dir=-1;dir<=1;dir+=2){
                            T_INDEX axis_vector=T_INDEX::Axis_Vector(axis)*dir;
                            T_INDEX neighbor_index=index+axis_vector;
                            const T& neighbor_mass=Object_mass(oid)(neighbor_index(0)-1)(neighbor_index(1)-1);
                            axis_normal+=dir*neighbor_mass;
                        }
                        normal(axis)=axis_normal;
                    }

                    const T result=-normal.Dot_Product(obj_V-other_V);
                    if(result>(T)0.){
                        const T cell_mass=ALL_mass(cur_cell(0))(cur_cell(1));
                        const TV cell_p=ALL_p(cur_cell(0))(cur_cell(1));
                        obj.barriers.Append(index);
                        obj.velocities.Append(cell_p/cell_mass);
                        marker(oid)(cur_cell(0))(cur_cell(1))=(T)1.;
                    }
                    /*
                            a --->       b ->
                            delta_V = a_V - b_V
                            delta_V.Dot_Product(a_normal) > 0  => collision

                    */
                }   
            }
        }      
    }
}
//######################################################################
// ALL_Rasterize
//######################################################################
template<class T> void MPM_Example<T,3>::
ALL_Rasterize()
{
    const int number_of_objects=objects.size();
    Array<Array<Array<Array<T> > > > Objects_volume=Array<Array<Array<Array<T> > > >(number_of_objects,Array<Array<Array<T> > >(counts(0),Array<Array<T> >(counts(1),Array<T>(counts(2),(T)0.))));
    Array<Array<Array<Array<T> > > > ALL_volume=Array<Array<Array<Array<T> > > >(number_of_objects,Array<Array<Array<T> > >(counts(0),Array<Array<T> >(counts(1),Array<T>(counts(2),(T)0.))));

    Array<Array<Array<Array<T> > > > Objects_mass=Array<Array<Array<Array<T> > > >(number_of_objects,Array<Array<Array<T> > >(counts(0),Array<Array<T> >(counts(1),Array<T>(counts(2),(T)0.))));
    Array<Array<Array<Array<T> > > > ALL_mass=Array<Array<Array<Array<T> > > >(number_of_objects,Array<Array<Array<T> > >(counts(0),Array<Array<T> >(counts(1),Array<T>(counts(2),(T)0.))));

    Array<Array<Array<Array<TV> > > > Objects_p=Array<Array<Array<Array<TV> > > >(number_of_objects,Array<Array<Array<TV> > >(counts(0),Array<Array<TV> >(counts(1),Array<TV>(counts(2),TV()))));
    Array<Array<Array<Array<TV> > > > ALL_p=Array<Array<Array<Array<TV> > > >(number_of_objects,Array<Array<Array<TV> > >(counts(0),Array<Array<TV> >(counts(1),Array<TV>(counts(2),TV()))));

    Array<Array<Array<Array<T> > > > markers=Array<Array<Array<Array<T> > > >(number_of_objects,Array<Array<Array<T> > >(counts(0),Array<Array<T> >(counts(1),Array<T>(counts(2),(T)0.))));
    const Grid<T,3> grid(counts,domain);
#pragma omp parallel for
    for(int tid_process=0;tid_process<threads;++tid_process){
        const Interval<int> thread_x_interval=ALL_x_intervals(tid_process);
        for(int tid_collect=0;tid_collect<threads;++tid_collect){
            Array<int>& index=ALL_particle_bins(tid_process,tid_collect);
            for(int i=0;i<index.size();++i){
                T_Particle& p=particles(index(i));T_INDEX& closest_cell=p.closest_cell_X;
                const Interval<int> relative_interval=Interval<int>(thread_x_interval.min_corner-closest_cell(0),thread_x_interval.max_corner-closest_cell(0));
                for(T_Cropped_Influence_Iterator iterator(T_INDEX(-1),T_INDEX(1),relative_interval,p,false);iterator.Valid();iterator.Next()){
                    T_INDEX relative_index=iterator.Index()+T_INDEX(1);
                    const T& p_volume=p.volume;
                    const T p_mass=p.mass; const TV p_V=p.V;
                    const T weight=iterator.Weight(); 

                    T_INDEX cell_index=iterator.Current_Cell()-T_INDEX(1);

                    int oid=p.oid;
                    Objects_mass(oid)(cell_index(0))(cell_index(1))(cell_index(2))+=weight*p_mass;
                    Objects_volume(oid)(cell_index(0))(cell_index(1))(cell_index(2))+=weight*p_volume;
                    Objects_p(oid)(cell_index(0))(cell_index(1))(cell_index(2))+=weight*p_mass*p_V;

                    for(int obj_id=0;obj_id<number_of_objects;++obj_id){
                        ALL_mass(obj_id)(cell_index(0))(cell_index(1))(cell_index(2))+=weight*p_mass;
                        ALL_volume(obj_id)(cell_index(0))(cell_index(1))(cell_index(2))+=weight*p_volume;
                        ALL_p(obj_id)(cell_index(0))(cell_index(1))(cell_index(2))+=weight*p_mass*p_V;
                    }
                }
            }
        }
    }

#pragma omp parallel for
    for(int oid=0;oid<number_of_objects;++oid){
        T_Object& obj=objects(oid);
        const auto& object_ALL_volume=ALL_volume(oid);
        const auto& object_ALL_mass=ALL_mass(oid);
        const auto& object_ALL_p=ALL_p(oid);
        const auto& Object_volume=Objects_volume(oid);
        const auto& Object_mass=Objects_mass(oid);
        const auto& Object_p=Objects_p(oid);
        auto& marker=markers(oid);
        obj.barriers.Clear();
        obj.velocities.Clear();
        for(int i=0;i<obj.particle_ids.size();++i){
            const int pid=obj.particle_ids[i];
            T_Particle& p=particles(pid);
            p.ALL_Update_Closest_Cell(grid);
            const T_INDEX& closest_cell=p.closest_cell_X;
            for(Range_Iterator<3> iterator(T_INDEX(-1),T_INDEX(1));iterator.Valid();iterator.Next()){
                T_INDEX cur_cell=closest_cell+iterator.Index()-T_INDEX(1);      // 0-index
                const T& cell_Object_volume=Object_volume(cur_cell(0))(cur_cell(1))(cur_cell(2));
                const T& cell_ALL_volume=object_ALL_volume(cur_cell(0))(cur_cell(1))(cur_cell(2));
                const T cell_Other_volume=cell_ALL_volume-cell_Object_volume;
                const T flag=marker(cur_cell(0))(cur_cell(1))(cur_cell(2));
                if(flag<(T).5 && cell_Other_volume>(T)0. && cell_Object_volume>(T)0.){
                    T_INDEX index=cur_cell+T_INDEX(1);      // 1-indexed
                    // Log::cout<<index<<std::endl;
                    const T& obj_mass=Object_mass(cur_cell(0))(cur_cell(1))(cur_cell(2));
                    const TV& obj_p=Object_p(cur_cell(0))(cur_cell(1))(cur_cell(2));
                    const TV obj_V=obj_p/obj_mass;

                    const T& all_mass=object_ALL_mass(cur_cell(0))(cur_cell(1))(cur_cell(2));
                    const TV& all_p=object_ALL_p(cur_cell(0))(cur_cell(1))(cur_cell(2));
                    
                    const T other_mass=all_mass-obj_mass;
                    const TV other_p=all_p-obj_p;
                    const TV other_V=other_p/other_mass;
                
                    TV normal=TV();
                    for(int axis=0;axis<3;++axis){
                        T axis_normal=(T)0.;
                        for(int dir=-1;dir<=1;dir+=2){
                            T_INDEX axis_vector=T_INDEX::Axis_Vector(axis)*dir;
                            T_INDEX neighbor_index=index+axis_vector;
                            const T& neighbor_mass=Object_mass(neighbor_index(0)-1)(neighbor_index(1)-1)(neighbor_index(2)-1);
                            axis_normal+=dir*neighbor_mass;
                        }
                        normal(axis)=axis_normal;
                    }
                    // if(normal.Norm()!=(T)0.) normal/=normal.Norm();
                    const T result=-normal.Dot_Product(obj_V-other_V);
                    if(result>(T)0.){
                        const T cell_mass=object_ALL_mass(cur_cell(0))(cur_cell(1))(cur_cell(2));
                        const TV cell_p=object_ALL_p(cur_cell(0))(cur_cell(1))(cur_cell(2));
                        obj.barriers.Append(index);
                        obj.velocities.Append(cell_p/cell_mass);
                    }
                    marker(cur_cell(0))(cur_cell(1))(cur_cell(2))=(T)1.;
                    
                    /*
                            a --->       b ->
                            delta_V = a_V - b_V
                            delta_V.Dot_Product(a_normal) > 0  => collision

                    */
                }   
            }

        }      
    }
}
//######################################################################
// Save_Grid_Info
//######################################################################
template<class T> void MPM_Example<T,2>::
Save_Grid_Info()
{
    auto flags=hierarchy->Channel(level,flags_channel);
    const Grid<T,2>& grid=hierarchy->Lattice(level);
    {
        std::string file=output_directory+"/Grid_Info/Interior_Cell_"+std::to_string(Base::current_frame)+"_dX_"+std::to_string(grid.dX(0))+".txt";
        std::FILE *fp = std::fopen(file.c_str(),"w");
        //output the energy
        for(Range_Iterator<2> iterator(T_INDEX(1),counts);iterator.Valid();iterator.Next()){
            T_INDEX cell_index=iterator.Index(); 
            if(flags(cell_index._data)&Cell_Type_Interior){
                TV cell_location=grid.Center(cell_index);
                fprintf(fp,"%f %f\n",cell_location(0),cell_location(1));}
        }
        std::fclose(fp);
    }

    {
        auto X0_x=hierarchy->Channel(level,X0_channels(0));            auto X0_y=hierarchy->Channel(level,X0_channels(1));
        auto X_x=hierarchy->Channel(level,position_channels(0));       auto X_y=hierarchy->Channel(level,position_channels(1));
        auto V_x=hierarchy->Channel(level,velocity_star_channels(0));  auto V_y=hierarchy->Channel(level,velocity_star_channels(1));
        std::string file=output_directory+"/Output_"+std::to_string(counts(0))+"x"+std::to_string(counts(1))+"/Grid_"+std::to_string(counts(0))+"x"+std::to_string(counts(1))+"_Frame_"+std::to_string(Base::current_frame)+".txt";
        std::FILE *fp = std::fopen(file.c_str(),"w");
        //output the energy
        for(Range_Iterator<2> iterator(T_INDEX(1),counts);iterator.Valid();iterator.Next()){
            T_INDEX cell_index=iterator.Index(); 
            if(flags(cell_index._data)&Cell_Type_Interior){
                auto data=cell_index._data;
                fprintf(fp,"%f %f %f %f %f %f\n",X0_x(data),X0_y(data),X_x(data),X_y(data),V_x(data),V_y(data));}
        }
        std::fclose(fp);
    }
}
//######################################################################
// Grid_Based_Collision
//######################################################################
template<class T> void MPM_Example<T,3>::
Spiral_Test(const T& time,const T& dt)
{  
    boat.Move(time,dt,particles);
}
//######################################################################
// Save_Grid_Info
//######################################################################
template<class T> void MPM_Example<T,3>::
Save_Grid_Info()
{
    auto flags=hierarchy->Channel(level,flags_channel);
    const Grid<T,3>& grid=hierarchy->Lattice(level);
    std::string file=output_directory+"/Grid_Info/Interior_Cell_"+std::to_string(Base::current_frame)+"_dX_"+std::to_string(grid.dX(0))+".txt";
    std::FILE *fp = std::fopen(file.c_str(),"w");
    //output the energy
    for(Range_Iterator<3> iterator(T_INDEX(1),counts);iterator.Valid();iterator.Next()){
        T_INDEX cell_index=iterator.Index(); 
        if(flags(cell_index._data)&Cell_Type_Interior){
            TV cell_location=grid.Center(cell_index);
            fprintf(fp,"%f %f %f\n",cell_location(0),cell_location(1),cell_location(2));}
    }
    std::fclose(fp); 
}
//######################################################################
// Save_Particle_Info
//######################################################################
template<class T> void MPM_Example<T,2>::
Save_Particle_Info()
{
    auto flags=hierarchy->Channel(level,flags_channel);
    {
        std::string file=output_directory+"/Particle_Info/Particle_Color_"+std::to_string(Base::current_frame)+".txt";
        std::FILE *fp = std::fopen(file.c_str(),"w");
        //output the energy
        for(auto& p: particles){
            const T& Je=p.constitutive_model.Je;
            fprintf(fp,"%f %f %f\n",p.X(0),p.X(1),Je);
        }
        std::fclose(fp);
    }
    {
        std::string file=output_directory+"/Output_"+std::to_string(counts(0))+"x"+std::to_string(counts(1))+"/Particle_"+std::to_string(counts(0))+"x"+std::to_string(counts(1))+"_Frame_"+std::to_string(Base::current_frame)+".txt";
        std::FILE *fp = std::fopen(file.c_str(),"w");
        //output the energy
        for(auto& p: particles){
            fprintf(fp,"%f %f %f %f %f %f\n",p.X0(0),p.X0(1),p.X(0),p.X(1),p.V(0),p.V(1));
        }
        std::fclose(fp);
    }

}
//######################################################################
// Save_Particle_Info
//######################################################################
template<class T> void MPM_Example<T,3>::
Save_Particle_Info()
{
    auto flags=hierarchy->Channel(level,flags_channel);
    std::string file=output_directory+"/Particle_Info/Particle_Color_"+std::to_string(Base::current_frame)+".txt";
    std::FILE *fp = std::fopen(file.c_str(),"w");
    //output the energy
    for(auto& p: particles){
        const T& Je=p.constitutive_model.Je;
        const T rgb=(Je+5.)/35.;
        fprintf(fp,"%f %f %f 0 200 %f\n",p.X(0),p.X(1),p.X(2),rgb*(T)255.);
    }
    std::fclose(fp);
}
//######################################################################
template class Nova::MPM_Example<float,2>;
template class Nova::MPM_Example<float,3>;
#ifdef COMPILE_WITH_DOUBLE_SUPPORT
template class Nova::MPM_Example<double,2>;
template class Nova::MPM_Example<double,3>;
#endif
