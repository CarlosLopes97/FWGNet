#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

void saveData() {
    FILE* f = fopen("data.txt", "w");
    if(f == NULL) {
        printf("cant save data");
        return;
    }
    //write some data (as integer) to file
    fprintf(f, "%d %d %d\n", 12, 45, 33);
    fprintf(f, "%d %d %d\n", 1, 2, 3);
    fprintf(f, "%d %d %d\n", 9, 8, 7);

    fclose(f);
}
using namespace std;

std::string** create(int rows, int columns)
	{
		std::string** table = new std::string*[rows];
		
		for(int i = 0; i < rows; i++) 
		{
			table[i] = new std::string[columns]; 
			for(int j = 0; j < columns; j++)
			{ 
			table[i][j] = "0"; 
			}// sample set value;    
		}
		return table;
	}

// std::string to_string(int x)
// {
//     std::ostringstream ss;
//     ss << x;
//     return ss.str();
// }

void loadData() 
{
    
    int n_lines = 2;

    char *get_proto;
    int get_ip_src;
    int get_ip_dst;

    int n_param = 3;
    std::string** proto_ip = create(n_lines, n_param);
    
    // string data1;
    
    char data1[4096];
    int data2, data3;
    FILE* f = fopen("/home/carl/New_Results/Files/proto_ips.txt", "r");
    // FILE* f = fopen("data.txt", "r");
    if(f == NULL) 
    {
        printf("cant open file");
    
    }
    else{
        int i = 0;
        // printf("ORIGINAL\n");
        while(fscanf(f, "%s %d %d", data1, &data2, &data3) == n_param) 
        {
            
            // printf("%s %d %d\n", data1, data2, data3);

            get_proto = data1;  
            if (get_proto)  
            proto_ip[i][0] = get_proto;
            
            get_ip_src = data2; 
            if (get_ip_src)  
            proto_ip[i][1] = to_string(get_ip_src);

            get_ip_dst = data3;
            if (get_ip_dst)  
            proto_ip[i][2] = to_string(get_ip_dst);

            i++;
        }
        fclose(f);
        

    }
    //load data from file, fscanf return the number of read data
    //so if we reach the end of file (EOF) it return 0 and we end
    // while(fin, "%s %s %d %d %d", data1, data2, &data3, &y, &z) {
    

    // std::string** proto_ip = create(n_lines, n_param);
    // printf("NEW\n");

    for (int i = 0; i < n_lines; ++i)
    {
        cout<<"PROTO: "<<proto_ip[i][0]<<" IP_SRC:"<<proto_ip[i][1]<<" IP_DST: "<<proto_ip[i][2]<<endl;

        string str = "eth";
        std::size_t found = proto_ip[i][0].find("kio");
        if (found!=std::string::npos){
            cout << str<< " also found at: " << found << "\n";
        }else{
            cout << str <<" not found at: \n";
            }
        // cout<<"PROTO: "<<endl;
    }
}

int main() {
    // saveData();
    // int vec_main [3];
    loadData();

    // for (int i = 0; i<2;i++){
    //     printf("data1 = %d", vec_main[i]);

    // }
    return 0;
}