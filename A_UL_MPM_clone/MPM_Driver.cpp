//!#####################################################################
//! \file MPM_Driver.cpp
//!#####################################################################
#include <chrono>
#include "MPM_Driver.h"
using namespace std::chrono;
using namespace Nova;
//######################################################################
// Constructor
//######################################################################
template<class T,int d> MPM_Driver<T,d>::
MPM_Driver(MPM_Example<T,d>& example_input)
    :Base(example_input),example(example_input)
{}
//######################################################################
// Initialize
//######################################################################
template<class T,int d> void MPM_Driver<T,d>::
Initialize()
{
    time=(T)0.;
    if(!example.restart) example.Initialize();
    else example.Read_Output_Files(example.restart_frame);
}
//######################################################################
// Test
//######################################################################
template<class T,int d> void MPM_Driver<T,d>::
Test()
{
}
//######################################################################
// Execute_Main_Program
//######################################################################
template<class T,int d> void MPM_Driver<T,d>::
Execute_Main_Program() 
{
    Initialize();
    if(example.vonMises)File_Utilities::Create_Directory(example.output_directory+"/von_mises");
    Simulate_To_Frame(example.last_frame);
}
//######################################################################
// Advance_To_Target_Time
//######################################################################
template<class T,int d> void MPM_Driver<T,d>::
Advance_To_Target_Time(const T target_time)
{
    T min_dt=(T)1e-6;
    T max_dt=(T).005;
    T cfl=example.cfl;
    T dx_min=example.hierarchy->Lattice(0).dX(0);
    bool done=false;
    for(int substep=1;!done;substep++){
        // if(example.cnt>=200){example.to_update=true;example.cnt=0;}
        // example.cnt++;
        // example.to_update=false;
        bool skip=false;
        T max_v=example.Max_Particle_Velocity();
        Log::Scope scope("SUBSTEP","substep "+std::to_string(substep));
        T dt=std::max(min_dt,std::min(max_dt,cfl*dx_min/std::max(max_v,(T)1e-2)));
        dt=example.const_dt;
        done=true;
        if(example.use_const_dt) dt=example.const_dt;
        if(target_time-time<dt*1.001){
            dt=target_time-time;
            done=true;
        }
        else if(target_time-time<(T)2.*dt){
            dt=(target_time-time)*(T).5;
        }
        if(dt<0) {
            skip=true;
            done=true;
        }
        if(!skip){
            example.Set_Dt(dt);
            dt=Advance_Step();
            total_dt+=dt;
            total_cnt++;
            example.cnt++;
            Log::cout<<"dt: "<<dt<<std::endl;
            // if(!done) example.Write_Substep("END Substep",substep,0);
            // done=true;
            time+=dt;
        }
        
        }
    example.Save_Grid_Info();
    example.Save_Particle_Info();
}
//######################################################################
// Simulate_To_Frame
//######################################################################
template<class T,int d> void MPM_Driver<T,d>::
Simulate_To_Frame(const int target_frame)
{
    example.frame_title="Frame "+std::to_string(example.current_frame);
    if(!example.restart){
        Write_Output_Files(example.current_frame);
        if(example.vonMises)example.Print_von_Mises_Stress(example.current_frame);
    }

    while(example.current_frame<target_frame){
        Log::Scope scope("FRAME","Frame "+std::to_string(++example.current_frame));

        Advance_To_Target_Time(example.Time_At_Frame(example.current_frame));

        example.frame_title="Frame "+std::to_string(example.current_frame);
        Write_Output_Files(++example.output_number);
        if(example.vonMises)example.Print_von_Mises_Stress(example.output_number);

        *(example.output)<<"TIME = "<<time<<std::endl;}
}
//######################################################################
// Advance_Step
//######################################################################
template<class T,int d> T MPM_Driver<T,d>::
Advance_Step()
{
    return example.Solve(time);
}
//######################################################################
template class Nova::MPM_Driver<float,2>;
template class Nova::MPM_Driver<float,3>;
#ifdef COMPILE_WITH_DOUBLE_SUPPORT
template class Nova::MPM_Driver<double,2>;
template class Nova::MPM_Driver<double,3>;
#endif
