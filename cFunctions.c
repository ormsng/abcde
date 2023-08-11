#include "Structs.h"
#include "Proto.h"


int printToOutputFile(char* str) 
{
    FILE *outputStream = fopen(OUTPUTFILE, "w"); 

    if (outputStream == NULL)
    {
        fprintf(stderr, "can't open %s\n", OUTPUTFILE);
        return 0;
    }
    if(str!=NULL){
        fprintf(outputStream, "%s\n", str);
    }

    fclose(outputStream);
    return 1;
}

int printResults(int indexes[3], double t)
{
    FILE *outputStream = fopen(OUTPUTFILE, "a"); 
    if (outputStream == NULL)
    {
        fprintf(stderr, "can't open %s\n", OUTPUTFILE);
        return 0;
    }
    int res = fprintf(outputStream, "Points pointID%d, pointID%d, pointID%d satesfy Proximity Criteria at t = %.02f\n", indexes[0], indexes[1], indexes[2], t);
    fclose(outputStream);
    return res;
}

int checkFlagsAndPrintOut(int N, int *globalFlags, double t)
{
    int indexes[3] = {-1, -1, -1};
    int counter = 0;
#pragma omp parallel for shared(counter) schedule(dynamic)
    for (int i = 0; i < N; i++)
    {
        if (globalFlags[i] == 1)
        {
            if (counter < 3)
            {
                indexes[counter++] = i;
            }
        }
    }
    if (indexes[2] != -1)
    { 
        if (!printResults(indexes, t))
            return -1;
        else
            return 1;
    }
    return 0;
}

Axis *readFile(int *N, int *K, float *D, int *Tcount)
{
    printf("reading input file...\n");
    FILE *inputStream = fopen(INPUTFILE, "r"); 

    if (inputStream == NULL)
    {
        fprintf(stderr, "can't open %s\n", INPUTFILE);
        return NULL;
    }

    if (fscanf(inputStream, "%d %d %f %d\n", N, K, D, Tcount) != 4)
    {
        fclose(inputStream);
        fprintf(stderr, "missing necessary input for the program\n");
        return NULL;
    }
    Axis *inputArr = (Axis *)malloc((*N) * sizeof(Axis));
    if (inputArr == NULL)
    {
        fclose(inputStream);
        fprintf(stderr, "allocation error - couldnwt allocate inputArr\n");
        return NULL;
    }
    for (int i = 0; i < (*N); i++)
    {
        int id;
        if (fscanf(inputStream, "%d ", &id) != 1 || id > *N) 
        {
            fclose(inputStream);
            fprintf(stderr, "reading error - missing index\n");
            free(inputArr);
            return NULL;
        }
        if (fscanf(inputStream, "%f %f %f %f\n", &inputArr[id].x1, &inputArr[id].x2, &inputArr[id].a, &inputArr[id].b) != 4)
        {
            fclose(inputStream);
            fprintf(stderr, "reading error - missing value at index %d\n", id);
            free(inputArr);
            return NULL;
        }
    }
    fclose(inputStream);
    printf("reading went successfully\n");
    return inputArr;
}

void freePointers(int numPointers, ...)
{
    va_list pointers;
    va_start(pointers, numPointers);

    for (int i = 0; i < numPointers; i++)
    {
        void *ptr = va_arg(pointers, void *);

        if (ptr != NULL)
        {
            free(ptr);
        }
    }

    va_end(pointers);
}
