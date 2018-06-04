#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <math.h>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <limits>
#include <time.h>

#include <fstream>
#include <vector>
#include <complex>

using namespace std;

#define PI 3.14159265358979323846
complex<double> J(0.0, 1);

#define DEBUG_ALL
#ifdef DEBUG_ALL
   #define DO_DEBUG_ALL(statement) statement
   #define DEBUG_BASIC
#else
   #define DO_DEBUG_ALL(statement)
#endif

//#define DEBUG_BASIC
#ifdef DEBUG_BASIC
   #define DO_DEBUG_BASIC(statement) statement
#else
   #define DO_DEBUG_BASIC(statement)
   #define DEBUG_DISABLED
#endif

#ifdef DEBUG_DISABLED
   #define DO_DEBUG_DISABLED(statement) statement
#else
   #define DO_DEBUG_DISABLED(statement)
#endif


template <class T>
void sinwave( std::complex<T> * table, int samples, int freq, int sampling_rate, int amp)
{
    for (int  i = 0; i < samples; ++i)
    {
       table[i] =  amp * sin(freq * (2* PI)*i/sampling_rate);
    }
}

unsigned int bitReverse(unsigned int num, unsigned int bits)
{
    unsigned int reverse_num = 0, i, temp;
 
    for (i = 0; i < bits; i++)
    {
        temp = (num & (1 << i));
        if(temp)
            reverse_num |= (1 << ((bits - 1) - i));
    }
  
    return reverse_num;
}

int main(int argc, char **argv)
{
    int rank;
    int comm_size;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    const int CORE_0 = 0;
    const int NUMBER_OF_CORES = atoi(argv[1]);
    const int NUMBER_OF_NODES = atoi(argv[2]); //pbs nodes
    const int NUMBER_OF_PPN = atoi(argv[3]);  //pbs ppn
    if (rank == CORE_0) printf("%d cores (%dx%d)\n", NUMBER_OF_CORES, NUMBER_OF_NODES, NUMBER_OF_PPN);
    const int CORE_LAST = NUMBER_OF_CORES - 1;

    int sampling_rate = atoi(argv[4]);
    // int sampling_rate = 1024;
    int signal_time_sec = atoi(argv[5]);
    int total_samples_taken = sampling_rate * signal_time_sec;
    

    //round up to power of 2
    int temp_samples = 1;
    while (temp_samples < total_samples_taken)
    {
        temp_samples = temp_samples << 1;
        //printf("%d ", temp_samples);
    }
    
    int total_samples = temp_samples;
    int samples_per_core = total_samples/NUMBER_OF_CORES;
    std::complex<double> * data = new std::complex<double>[total_samples];
    std::complex<double> * data_reversed = new std::complex<double>[total_samples];
    std::complex<double> * fft_outcome = new std::complex<double>[total_samples];

    std::complex<double> * chunk_data_reversed = new std::complex<double>[samples_per_core];
    std::complex<double> * chunk_fft_outcome = new std::complex<double>[samples_per_core];
    std::complex<double> * neighbour_data = new std::complex<double>[samples_per_core];
    
    int total_bits =  log2(total_samples);
    
    //generate signal
    if(rank == CORE_0)
    {
        sinwave(data, total_samples, 1, sampling_rate , 1);

        //bit reverse
        for(int i = 0; i < total_samples; ++i)
        {
            unsigned int reversed_index = bitReverse(i, total_bits);
            data_reversed[i] = data[reversed_index];
            fft_outcome[i] =  data[reversed_index];
            
            //printf("\n%d - %d", i, reversed_index);
            //printf("\n%f", data[i]);
        }
    }

    //scatter data
    MPI_Scatter(data_reversed, samples_per_core, MPI_C_DOUBLE_COMPLEX,
                chunk_data_reversed, samples_per_core, MPI_C_DOUBLE_COMPLEX, CORE_0, MPI_COMM_WORLD);

    //and copy it to chunk_fft_outcome
    memcpy(chunk_fft_outcome, chunk_data_reversed, samples_per_core * sizeof(std::complex<double>));


    double start_time, algorithm_time, end_time;
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == CORE_0)
    {
        start_time = MPI_Wtime();
    }
  

    //wk coeff
    std::complex<double> * wk =  new std::complex<double>[total_samples];
    for (int i = 0; i < total_samples; ++i)
    {
        wk[i] = exp(-J *(2 * PI * i / total_samples));
        //printf("\n%f %fi", std::real(wk[i]),std::imag(wk[i]));
    }
    
    //fft
    MPI_Barrier(MPI_COMM_WORLD);
    for(int step = 1 ; step <= total_bits; ++step)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        int power_of_step = pow(2, step);

        for(int sample_id = 0; sample_id < samples_per_core; ++sample_id)
        {
            int sample_mod_power_of_step = (sample_id+(rank*samples_per_core))%power_of_step;
            if(sample_mod_power_of_step >= power_of_step / 2)
            {
                chunk_data_reversed[sample_id] *= wk[(total_samples/power_of_step) * (sample_mod_power_of_step-power_of_step/2)]; 
                //printf("\n%d core: %d element multiplied by %d wk element", rank,(sample_id+(rank*samples_per_core)),total_samples/power_of_step * (sample_mod_power_of_step-power_of_step/2));
            }
        }

        if (power_of_step <= samples_per_core)
        {           
            for(int sample_id = 0; sample_id < samples_per_core; ++sample_id)
            {
                int sample_mod_power_of_step = sample_id%power_of_step;
                if(sample_mod_power_of_step >= power_of_step / 2)
                {
                    chunk_fft_outcome[sample_id] =  chunk_data_reversed[sample_id  - power_of_step / 2] - chunk_data_reversed[sample_id];
                    //printf("\n%d subadd %d element", sample_id + (rank*samples_per_core),  sample_id + (rank*samples_per_core)  - power_of_step / 2);
                }
                else
                {
                    chunk_fft_outcome[sample_id] += chunk_data_reversed[sample_id  + power_of_step / 2];
                    //printf("\n%d add %d element", sample_id + (rank*samples_per_core), sample_id + (rank*samples_per_core)  + power_of_step / 2);
                }
            }
        }
        else
        {
            int neighbour;
            int power_of_core_step = power_of_step / samples_per_core;
            int sample_mod_power_of_core_step = rank%power_of_core_step;

            if(sample_mod_power_of_core_step >= power_of_core_step / 2)
            {
                neighbour = rank - power_of_core_step / 2;

                MPI_Send(chunk_data_reversed, samples_per_core, MPI_C_DOUBLE_COMPLEX, neighbour, 0, MPI_COMM_WORLD);
                MPI_Recv(neighbour_data, samples_per_core, MPI_C_DOUBLE_COMPLEX,  neighbour, 0, MPI_COMM_WORLD, &status);
                //printf("\n%d core: received samples: %d-%d", rank, ((neighbour*samples_per_core)), (((neighbour+1)*samples_per_core-1)));
            }
            else
            {
                neighbour = rank + power_of_core_step / 2;
                MPI_Recv(neighbour_data, samples_per_core, MPI_C_DOUBLE_COMPLEX,  neighbour, 0, MPI_COMM_WORLD, &status);
                MPI_Send(chunk_data_reversed, samples_per_core, MPI_C_DOUBLE_COMPLEX, neighbour, 0, MPI_COMM_WORLD);
                //printf("\n%d core: received samples: %d-%d", rank, ((neighbour*samples_per_core)), (((neighbour+1)*samples_per_core-1)));
            }

            for(int sample_id = 0; sample_id < samples_per_core; ++sample_id)
            {
                if(sample_mod_power_of_core_step >= power_of_core_step / 2)
                {
                    chunk_fft_outcome[sample_id] = neighbour_data[sample_id] - chunk_data_reversed[sample_id];
                    //printf("\n%d core: %d element - subadd %d element", rank,sample_id + (rank*samples_per_core), (sample_id + (neighbour*samples_per_core)));
                }
                else
                {
                    chunk_fft_outcome[sample_id] += neighbour_data[sample_id];
                     //printf("\n%d core: %d element - add %d element", rank,sample_id + (rank*samples_per_core), (sample_id + (neighbour*samples_per_core)));
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        memcpy(chunk_data_reversed, chunk_fft_outcome, samples_per_core * sizeof(std::complex<double>));
        // for(int i = 0; i < samples_per_core; ++i)
        // {
        //     double result = sqrt(pow(std::real(chunk_fft_outcome[i]),2) + pow(std::imag(chunk_fft_outcome[i]),2));
        //     //printf("\n%d core: point %d %f", rank,i,result);
        // }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == CORE_0)
    {
        algorithm_time = MPI_Wtime();
    }
    
    //std::string file_name =  argv[2];
    //read file to get number of data
    //std::fstream ifile(file_name.c_str(), std::ios_base::in);


    //pass all data
    //MPI_Scatter(all_points, DATA_PER_CORE, MPI_2INT,
    //             chunk_points, DATA_PER_CORE, MPI_2INT, CORE_0, MPI_COMM_WORLD);

    //broadcast initial centroids x
    //MPI_Bcast(centroids, CENTROIDS, MPI_2INT, CORE_0, MPI_COMM_WORLD);


    //MPI_Barrier(MPI_COMM_WORLD);
    //float start_time = MPI_Wtime();

    //MPI_Barrier(MPI_COMM_WORLD);
    //float duration = MPI_Wtime() - start_time;
    //if(rank == CORE_0)
    //{
        //printf("%d: execution time - %f\n", rank ,duration);
        //for(int i = 0; i< CENTROIDS; ++i)
        //{
            //printf("%d,%d\n",centroids[i].x, centroids[i].y);
        //
    //}

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(chunk_fft_outcome, samples_per_core, MPI_C_DOUBLE_COMPLEX,
               fft_outcome, samples_per_core, MPI_C_DOUBLE_COMPLEX, CORE_0, MPI_COMM_WORLD);

    double* results = (double *) calloc(total_samples, sizeof(double));
    if(rank == CORE_0)
    {
        for(int i = 0; i < total_samples; ++i)
        {
            results[i] = sqrt(pow(std::real(fft_outcome[i]),2) + pow(std::imag(fft_outcome[i]),2));
            // printf("\n%f", result);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == CORE_0)
    {
        end_time = MPI_Wtime();
        printf("\n");

        printf("sampling_rate: %d | signal_time_sec: %d\n", sampling_rate, signal_time_sec);
        printf("total_samples_taken: %d | total samples: %d\n", total_samples_taken, total_samples);
        printf("algorithm time: %f\n", algorithm_time - start_time);
        printf("total time: %f\n", end_time - start_time);

        // printf("\n\n=================== RESULTS ===================");
        
        // for(int i = 0; i < total_samples; ++i)
        // {
        //     // printf("\n%f | %f | %fi", results[i], std::real(fft_outcome[i]), std::imag(fft_outcome[i]));
        //     printf("\n%f", results[i]);
        // }

        // printf("\n\n\n\n=================== DATA ===================");

        // for(int i = 0; i < total_samples; ++i)
        // {
        //     printf("\n%f", data[i]);
        // }
    }

    MPI_Finalize();
    delete[] data;
    delete[] data_reversed;
    delete[] fft_outcome;
    delete[] chunk_data_reversed;
    delete[] chunk_fft_outcome;
    delete[] neighbour_data;
    
   return 0;

}
