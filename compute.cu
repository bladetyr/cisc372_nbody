#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
__global__ void compute(){
	int i,j,k;
	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);

	//cuda versions of values and accels
	cudaMalloc((void**)&dValues, sizeof(float)*NUMENTITIES*NUMENTITIES);
	cudaMalloc((void**)&dAccels, sizeof(float)*NUMENTITIES);
	//copy those to run on GPU
	cudaMemcpy(dValues, values, sizeof(float)*NUMENTITIES*NUMENTITIES, cudaMemcpyHostToDevice);
	cudeMemcpy(dAccels, accels, sizeof(float)*NUMENTITIES, cudaMemcpyHostToDevice);

	//make an acceleration matrix which is NUMENTITIES squared in size;
	for (i=0;i<NUMENTITIES;i++)
		dAccels[i]=&dValues[i*NUMENTITIES];
	//first compute the pairwise accelerations.  Effect is on the first argument.
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	for (i;i<NUMENTITIES;i++){
		for (j;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(dAccels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(dAccels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}

	//reset i and j values for these for loops
	i = blockIdx.x * blockDim.x + threadIdx.x;
        j = blockIdx.y * blockDim.y + threadIdx.y;

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=dAccels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]=hVel[i][k]*INTERVAL;
		}
	}
	free(accels);
	free(values);
	cudaFree(dValues);
	cudaFree(dAccels);
}
