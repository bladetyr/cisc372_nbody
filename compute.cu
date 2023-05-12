#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

//separating compute() into cuda functions
__global__ void accelMatrix(vector3 values, vector3* accels, vector3 d_hVel, vector3 d_hPos, double d_mass){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	for (int i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
	//first compute the pairwise accelerations.  Effect is on the first argument.
	int i = threadIdx.x;
	//int j = blockIdx.y * blockDim.y + threadIdx.y;
	int j = 0;
	int k;
	for (i;i<NUMENTITIES;i++){
		for (j;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=d_hPos[i][k]-d_hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}
}

__global__ void sumMatrix(vector3 d_hVel, vector3 d_hPos, vector3 accels){
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	int i = threadIdx.x;
	//int j = blockIdx.y * blockDim.y + threadIdx.y;
	int j = 0;
	int k;
	if(i < NUMENTITIES) {
		vector3 accel_sum={0,0,0};
		for (j;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			d_hVel[i][k]+=accel_sum[k]*INTERVAL;
			d_hPos[i][k]=d_hVel[i][k]*INTERVAL;
		}
	}
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	vector3* dValues;
	vector3** dAccels;
	double* d_mass;

	//cuda versions of values and accels
	cudaMalloc((void**)&dValues, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	cudaMalloc((void**)&dAccels, sizeof(vector3)*NUMENTITIES);
	//copy those to run on GPU
	//cudaMemcpy(dValues, dValues, sizeof(float)*NUMENTITIES*NUMENTITIES, cudaMemcpyHostToDevice);
	//cudaMemcpy(dAccels, dAccels, sizeof(float)*NUMENTITIES, cudaMemcpyHostToDevice);
	//copy the global variables too
	cudaMalloc((void**)&d_hVel, sizeof(double)*NUMENTITIES);
	cudaMalloc((void**)&d_hPos, sizeof(double)*NUMENTITIES);
	cudaMalloc((void**)&d_mass, sizeof(double)*NUMENTITIES);

	cudaMemcpy(d_hVel, hVel, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hPos, hPos, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(double), cudaMemcpyHostToDevice);

	accelMatrix<<<1,1>>>(dValues, dAccels, *d_hVel, *d_hPos, *d_mass);
	cudaDeviceSynchronize();
	sumMatrix<<<1,1>>>(*d_hVel, *d_hPos, *dAccels);
	cudaDeviceSynchronize();
	//free(accels);
	//free(values);
	
	cudaFree(dAccels);
	cudaFree(dValues);
	cudaFree(d_mass);
	cudaFree(d_hVel);
	cudaFree(d_hPos);
}
