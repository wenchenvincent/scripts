#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc!=3) {
    	cout << "Number of Argument is " << argc << endl;
	cout << "It should be 2" << endl;
	exit(-1);
    }
    
    char* p;
    long num_bytes = strtol(argv[1], &p, 10);
    char buffer[num_bytes];
    FILE * filp = fopen(argv[2], "rb"); 
    int bytes_read = fread(buffer, sizeof(char), num_bytes, filp);
    fclose(filp);
    cout << "Number of bytes read is " << bytes_read << endl;
    if (bytes_read != num_bytes) {
    	cout << "Incorrect number of bytes read is " << bytes_read << endl;
    }
    FILE* fo = fopen(strcat(argv[2], ".txt"), "w");
    half* hp = reinterpret_cast<half*>(&buffer[0]);
    for (int i=0;i<num_bytes/sizeof(half);i++) {
    //   printf("%d %x = %f\n", i, *(short*)&hp[i], float(hp[i])); 
       if (float(hp[i]) != float(hp[i]))
         printf("%d %x = %f\n", i, *(uint16_t*)&hp[i], float(hp[i])); 

       fprintf(fo, "%d %x = %f\n", i, *(uint16_t*)&hp[i], float(hp[i])); 
    }
    fclose(fo);

}

