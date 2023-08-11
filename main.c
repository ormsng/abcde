#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include "Structs.h"
#include "mpiHelper.h"
#include "Proto.h"

#define MASTER 0
int main(int argc, char *argv[])
{
   int size, rank;
   int N;                   
   int K;                    
   int tCount;              
   float D;                 
   Axis *data = NULL;      
   Axis *axisChunk = NULL;  
   Point *allPoints = NULL; 
   Point *pointChunk = NULL;
   int *globalFlags = NULL;  
   int *flags = NULL;        
   double start;             
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   MPI_Datatype MPI_Axis;
   createMPIaxis(&MPI_Axis);
   MPI_Datatype MPI_Point;
   createMPIpoint(&MPI_Point);

   if (rank == MASTER)
   {
      data = readFile(&N, &K, &D, &tCount);
      if (data == NULL)
      {
         MPI_Abort(MPI_COMM_WORLD, 1);
      }
      if (!printToOutputFile(NULL))
      {
         freePointers(1, data); 
         MPI_Abort(MPI_COMM_WORLD, 1);
      }
      start = MPI_Wtime();
   }
   int counts[4] = {1, 1, 1, 1};
   MPI_Datatype myTypes[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT};
   _MPI_severalBcast(MASTER, MPI_COMM_WORLD, counts, myTypes, 4, &N, &K, &tCount, &D); 
   int chunkSize = N / size;
   axisChunk = (Axis *)malloc(sizeof(Axis) * chunkSize);
   pointChunk = (Point *)malloc(sizeof(Point) * chunkSize);
   allPoints = (Point *)malloc(N * sizeof(Point));
   if (axisChunk == NULL || pointChunk == NULL || allPoints == NULL)
   {
      freePointers(4, data, axisChunk, pointChunk, allPoints); 
      MPI_Abort(MPI_COMM_WORLD, 2);
   }
   MPI_Scatter(data, chunkSize, MPI_Axis, axisChunk, chunkSize, MPI_Axis, MASTER, MPI_COMM_WORLD);
   flags = (int *)malloc(chunkSize * sizeof(int));
   if (rank == MASTER)
   {
      globalFlags = (int *)malloc(N * sizeof(int));
   }
   int problematicIntervals = 0, i; 
   for (i = 0; i <= tCount; i++) 
   {
      double t = 2.0 * i / tCount - 1;
      if(rank==MASTER)
      {
         printf("i = %d \t t = %.02f   ",i,t); 
      } 
      if (!computePointsOnGPU(axisChunk, pointChunk, chunkSize, t)) 
      {
         freePointers(6, data, axisChunk, pointChunk, allPoints, globalFlags, flags); 
         MPI_Abort(MPI_COMM_WORLD, 3);
      }
      MPI_Allgather(pointChunk, chunkSize, MPI_Point, allPoints, chunkSize, MPI_Point, MPI_COMM_WORLD); 
      checkProximityCriteriaOnGPU(rank, allPoints, N, flags, chunkSize, D, K); 

      MPI_Gather(flags, chunkSize, MPI_INT, globalFlags, chunkSize, MPI_INT, MASTER, MPI_COMM_WORLD); 
      if (rank == MASTER)
      {
         int valid = checkFlagsAndPrintOut(N, globalFlags, t); 
         if (valid == -1)
         {
            freePointers(6, data, axisChunk, pointChunk, allPoints, globalFlags, flags); 
            MPI_Abort(MPI_COMM_WORLD, 4);
         }
         else{
            problematicIntervals+=valid; 
            printf("%s\n",valid? "found!" :" ");
         }

      } 
      MPI_Bcast(&problematicIntervals, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
   }
   if(rank==MASTER){
      double seconds = MPI_Wtime()-start;
      printf("Computation is done - %d Proximity Criteria satisfaction found in %02d:%02d minutes\n avg. time for epoch - %.02f seconds\n",
      problematicIntervals,((int)seconds)/60,((int)seconds)%60,seconds/i);
      if(problematicIntervals==0)
         printToOutputFile((char*)"There were no 3 points found at any t");
         
   }
   
   MPI_Type_free(&MPI_Axis);
   MPI_Type_free(&MPI_Point);
   freePointers(6, data, axisChunk, pointChunk, allPoints, globalFlags, flags);
   return 0;
}
