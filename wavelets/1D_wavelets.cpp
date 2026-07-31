
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

template<typename T>
void wavelet_forward(int size, T *s)
{
    int n = size;
    assert(size%2 == 0 && "size is not even\n");

    int n2 = n >> 1;
    
    // split even and odd samples
    T *temp = (T*)malloc(n2 * sizeof(T));
    T *d = s + n2;
    for (int k = 0; k < n2; k++)
        temp[k] = s[2 * k + 1];
    for (int k = 0; k < n2; k++)
        s[k] = s[2 * k];
    for (int k = 0; k < n2; k++)
        d[k] = temp[k];
    free(temp);

    // The layout is now *s* ... even values ... *d* ... odd values ...

    // First step: details -= samples
    for(int k=0;k<n2;k++)
        d[k] -= s[k];
    
    // Second step: samples += details/2
    for(int k=0;k<n2;k++)
        s[k] += d[k]/2;
    
    // Third step: lifting of details
    for(int k=1;k<n2-1;k++)
        d[k] += (s[k-1]-s[k+1])/4;
    // Edge cases
    d[0] += 0.75*s[0] - 1.0*s[1] + 0.25*s[2];
    d[n2-1] += 0.25*s[n2-1] - 1.0*s[n2-2] + 0.75*s[n2-3];
}

template<typename T>
void wavelet_inverse(int size, T *s)
{
    int n = size;
    assert(size%2 == 0 && "size is not even\n");

    int n2 = n >> 1;

    T *d = s + n2;

    // The layout is now *s* ... even values ... *d* ... odd values ...

    // First step: unlifting of details
    for(int k=1;k<n2-1;k++)
        d[k] -= (s[k-1]-s[k+1])/4;
    // Edge cases
    d[0] -= 0.75*s[0] - 1.0*s[1] + 0.25*s[2];
    d[n2-1] -= 0.25*s[n2-1] - 1.0*s[n2-2] + 0.75*s[n2-3];

    // Second step: samples -= details/2
    for(int k=0;k<n2;k++)
        s[k] -= d[k]/2;
    
    // Third step: details += samples
    for(int k=0;k<n2;k++)
        d[k] += s[k];

    // unsplit even and odd samples
    T *temp = (T*)malloc(n2 * sizeof(T));
    for (int k = 0; k < n2; k++)
    {
        temp[k] = d[k];
    }
    for (int k = n2-1; k > 0; k--)
    {
        s[2 * k] = s[k];
    }
    for (int k = 0; k < n2; k++)
        s[2 * k + 1] = temp[k];
    free(temp);
}

int main()
{
    const int size = 16;
    float data[size] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

    printf("Original data:\n");
    for (int i = 0; i < size; i++)
        printf("%f ", data[i]);
    printf("\n");

    wavelet_forward(size, data);

    printf("Transformed data:\n");
    for (int i = 0; i < size; i++)
        printf("%f ", data[i]);
    printf("\n");

    wavelet_inverse(size, data);

    printf("Reconstructed data:\n");
    for (int i = 0; i < size; i++)
        printf("%f ", data[i]);
    printf("\n");

    return 0;
}
