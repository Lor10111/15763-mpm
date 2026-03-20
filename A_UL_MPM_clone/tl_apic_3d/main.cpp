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
    high_resolution_clock::time_point tb = high_resolution_clock::now();
    enum {d=3};
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

    MPM_Driver<T,d> driver(*example);
    driver.Execute_Main_Program();
    Log::cout<<"average timestep: "<<driver.total_dt/driver.total_cnt<<std::endl;
    
    Log::cout<<"total update rt: "<<example->update_rt<<std::endl;
    Log::cout<<"total update cnt: "<<example->update_cnt<<std::endl;

    delete example;

    return 0;
}
