//!#####################################################################
//! \file main.cpp
//!#####################################################################
#include <chrono>
#include "../MPM_Driver.h"
#include "Standard_Tests/Standard_Tests.h"
using namespace Nova;
using namespace std::chrono;

int main(int argc,char** argv)
{
    Log::cout.precision(20);
    high_resolution_clock::time_point tb = high_resolution_clock::now();
    enum {d=2};
    typedef float T;
    MPM_Example<T,d> *example=new Standard_Tests<T,d>();
    example->Parse(argc,argv);
    example->bbox=Range<T,d>(example->domain.max_corner,example->domain.min_corner);
    File_Utilities::Create_Directory(example->output_directory);
    File_Utilities::Create_Directory(example->output_directory+"/common");
    Log::Instance()->Copy_Log_To_File(example->output_directory+"/common/log.txt",false);

    std::string s="command = ";
    for(int i=0;i<argc;i++)
    {
        s+=argv[i];
        s+=' ';
    }
    s+="\nworking directory = "+File_Utilities::Get_Working_Directory()+"\n";
    Log::cout<<s<<std::endl;

    // rt_total=(T)0.,rt_contact=(T)0.,rt_init_spgrid=(T)0.,rt_update_w=(T)0.,rt_group=(T)0.,rt_ras=(T)0.,rt_ras_vox=(T)0.;
    // T rt_update_consti=(T)0.,rt_exp_force=(T)0.,rt_boundary=(T)0.,rt_implicit=(T)0.,rt_g2p=(T)0.;
    MPM_Driver<T,d> driver(*example);
    driver.Execute_Main_Program();
    Log::cout<<"average timestep: "<<driver.total_dt/driver.total_cnt<<std::endl;
    Log::cout<<"################ Statistics ################"<<std::endl;   
    const T total=example->rt_total/example->cnt_total;
    const T pop=example->rt_pop/example->cnt_pop;
    const T contact=example->rt_contact/example->cnt_contact;
    const T init_spgrid=example->rt_init_spgrid/example->cnt_init_spgrid;
    const T update_w=example->rt_update_w/example->cnt_update_w;
    const T group=example->rt_group/example->cnt_group;
    const T ras=example->rt_ras/example->cnt_ras;
    const T ras_vox=example->rt_ras_vox/example->cnt_ras_vox;
    const T update_consti=example->rt_update_consti/example->cnt_update_consti;
    const T exp_force=example->rt_exp_force/example->cnt_exp_force;
    const T boudary=example->rt_boundary/example->cnt_boundary;
    const T imp=example->rt_implicit/example->cnt_implicit;
    const T g2p=example->rt_g2p/example->cnt_g2p;
    Log::cout<<"total: "<<total<<std::endl;
    Log::cout<<"pop: "<<pop<<": "<<pop/total*(T)100.<<"%"<<std::endl;
    Log::cout<<"contact: "<<contact<<": "<<contact/total*(T)100.<<"%"<<std::endl;
    Log::cout<<"init spgrid: "<<init_spgrid<<": "<<init_spgrid/total*(T)100.<<"%"<<std::endl;
    Log::cout<<"update weight: "<<update_w<<": "<<update_w/total*(T)100.<<"%"<<std::endl;
    Log::cout<<"group: "<<group<<": "<<group/total*(T)100.<<"%"<<std::endl;
    Log::cout<<"ras: "<<ras<<": "<<ras/total*(T)100.<<"%"<<std::endl;
    Log::cout<<"ras voxel: "<<ras_vox<<": "<<ras_vox/total*(T)100.<<"%"<<std::endl;
    Log::cout<<"update_constitutive: "<<update_consti<<": "<<update_consti/total*(T)100.<<"%"<<std::endl;
    Log::cout<<"exp_force: "<<exp_force<<": "<<exp_force/total*(T)100.<<"%"<<std::endl;
    Log::cout<<"boundary: "<<boudary<<": "<<boudary/total*(T)100.<<"%"<<std::endl;
    Log::cout<<"implicit: "<<imp<<": "<<imp/total*(T)100.<<"%"<<std::endl;
    Log::cout<<"G2P: "<<g2p<<": "<<g2p/total*(T)100.<<"%"<<std::endl;
    Log::cout<<"############################################"<<std::endl;   
    delete example;

    return 0;
}
