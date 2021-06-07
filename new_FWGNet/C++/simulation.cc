#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>

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

std::string** str_create(int rows, int columns)
	{
		std::string** table = new std::string*[rows];
		
		for(int i = 0; i < rows; i++) 
		{
			table[i] = new std::string[columns]; 
			for(int j = 0; j < columns; j++)
			{ 
			table[i][j] = "1"; 
			}// sample set value;    
		}
		return table;
	}


double** create(int rows, int columns)
	{
		double** table = new double*[rows];
		
		for(int i = 0; i < rows; i++) 
		{
			table[i] = new double[columns]; 
			for(int j = 0; j < columns; j++)
			{ 
			table[i][j] = 0; 
			}// sample set value;    
		}
		return table;
	}

// std::string to_string(int arr_Times)
// {
//     std::ostringstream ss;
//     ss << arr_Times;
//     return ss.str();
// }

void loadData() 
{
    
    int n_lines = 2;

    char *get_proto;
    int get_ip_src;
    int get_ip_dst;

    int n_net = 3;
    std::string** proto_ip = str_create(n_lines, n_net);
    
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
        while(fscanf(f, "%s %d %d", data1, &data2, &data3) == n_net) 
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
    // while(fin, "%s %s %d %d %d", data1, data2, &data3, &arr_Sizes, &z) {
    

    // std::string** proto_ip = create(n_lines, n_net);
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



int count_Lines(int num_Lines, std::string dir)
{
    // int aNumOfLines = 0;
    // ifstream aInputFile(dir.c_str()); 

    // string aLineStr;
    // while (getline(aInputFile, aLineStr))
    // {
    //     if (!aLineStr.empty())
    //         aNumOfLines++;
    // }
    // std::ifstream inFile(dir.c_str(), "r"); 
    // std::count(std::istreambuf_iterator<char>(inFile), 
    //          std::istreambuf_iterator<char>(), '\n');
    string line;
    std::ifstream myfile(dir.c_str());
    
    if (myfile.is_open())
    {
        while (getline(myfile,line) )
        {
        num_Lines++;
        }
        myfile.close();
        
    }

    else cout << "Unable to open file"; 

    return num_Lines;
    // return aNumOfLines;
}

void read_RV(void){

    int n_param = 3;
    int m_n_lines = 3;

    std::string** proto_ip = str_create(m_n_lines, n_param);
    proto_ip[0][0] += "eth:tcp:http:data-text-lines";

    std::string dir_size = "/home/carl/New_Results/Files/";
    dir_size += proto_ip[0][0];
    dir_size += "_size.txt";

    std::string dir_time = "/home/carl/New_Results/Files/";
    dir_time += proto_ip[0][0];
    dir_time += "_time.txt";

    int n_lines_Time;
    int n_lines_Size;
    int num_Lines = 3;

    n_lines_Time = count_Lines(num_Lines, dir_time);
    n_lines_Size = count_Lines(num_Lines,dir_size);

    int n_net = n_param;
    std::string** arr_Times = str_create(n_lines_Time, n_net);
    std::string** arr_Sizes = str_create(n_lines_Size, n_net);

    FILE *arq_Time;
    FILE *arq_Size;
    char row_Time[n_lines_Time];
    char row_Size[n_lines_Size];
    char *aux_Time;
    char *aux_Size;
    char *res_Time;
    char *res_Size;

    // Abre um arquivo TEXTO para LEITURA
    arq_Time = fopen(dir_time.c_str(), "rb"); 
    arq_Size = fopen(dir_size.c_str(), "rb");
    if (arq_Size == NULL)  // Se houve erro na abertura
    {
        cout<<"cant open file "<<dir_size<<endl;
    }
    if (arq_Time == NULL)  // Se houve erro na abertura
    {
        cout<<"cant open file "<<dir_time<<endl;
    }
  
    
    for(int i = 0; i< n_net; ++i)
    {
        arr_Sizes[0][i] = proto_ip[i][0];
        for (int j = 1; j<(n_lines_Size); ++j)
        {
            // Lê uma linha (inclusive com o '\n')
            res_Size = fgets(row_Size, n_lines_Size, arq_Size);  // Ler os caracteres ou até '\n'
            if (res_Size)  // Se foi possível ler
            aux_Size = row_Size;
            arr_Sizes[j][0] = aux_Size;
        }
    }
    for(int i = 0; i< n_net; ++i)
    {
        arr_Times[0][i] = proto_ip[i][0];
        for (int j = 1; j<(n_lines_Time); ++j)
        {
            // Lê uma linha (inclusive com o '\n')
            res_Time = fgets(row_Time, n_lines_Time, arq_Time);  // Ler os caracteres ou até '\n'
            if (res_Time)  // Se foi possível ler
            aux_Time = row_Time;
            arr_Times[j][0] = aux_Time;
        }
    }

    fclose (arq_Size);
    fclose (arq_Time);
}


void myapp()
{
    //                  |-------- = --------|
    //                  v                   v
    //       row_Size  n_net   m_n_file  n_m_param
    //              |  |              |  |
    // if (arr_Size[0][j] == proto_ip[i][0])

    // std::size_t m_found;
    // m_found = proto_ip[0][0].find("eth");
    
    // // int m_n_nodes = stoi(m_proto_ip[i][j]);
    // // m_container = std::string(m_proto_ip[i][0]);
    // // NetDeviceContainer m_device;
    // if (m_found!=std::string::npos)
    int m_n_param = 3;
    int m_n_file = 3;
    std::string** proto_ip = str_create(m_n_file, m_n_param);
    std::string** arr_Times = str_create(m_n_file, m_n_param);
    std::string** arr_Sizes = str_create(m_n_file, m_n_param);
    std::size_t m_found;
    std::string **size_rows_Size = str_create(m_n_file, m_n_param);
    size_rows_Size[1][1] += "20";
    size_rows_Size[2][1] += "50";
    std::string **size_rows_Time = str_create(m_n_file, m_n_param);
    size_rows_Time[1][1] += "30";
    size_rows_Time[2][1] += "10";
    // size_rows_Size[1][3] += "100";
    double interval = 0;
    double size_pckts = 0;
    for(int i = 0; i<m_n_param; ++i){
        for(int j = 1; j<m_n_file; ++j){
            // m_found = proto_ip[0][0].find(arr_Sizes[0][j].c_str());

            // get_interval
            
            // proto_ip, size_rows_Size
            // eth        500
            // eth
            
            
            
            
            if (arr_Sizes[0][j] == proto_ip[i][0] && size_rows_Size[j][0] == proto_ip[i][0])
            {   
                size_rows_Size[j][1] = std::to_string(std::stod(size_rows_Size[j][1])-1);
                
                interval = std::stod(arr_Sizes[i][j]);

                cout<<proto_ip[i][0]<<endl; 
                cout<<size_rows_Size[j][1]<<endl; 
                cout<<interval<<endl; 
            }


            if (arr_Times[0][j] == proto_ip[i][0] && size_rows_Time[j][0] == proto_ip[i][0])
            {   
                size_rows_Time[j][1] = std::to_string(std::stod(size_rows_Time[j][1])-1);
                
                size_pckts = std::stod(arr_Times[i][j]);
                
                cout<<proto_ip[i][0]<<endl; 
                cout<<size_rows_Time[j][1]<<endl; 
                cout<<size_pckts<<endl; 
            }

        }


    }

}


int main() {
    // saveData();
    // int vec_main [3];
    // loadData();
    read_RV();
    myapp();
    
    // for (int i = 0; i<2;i++){
    //     printf("data1 = %d", vec_main[i]);

    // }
    return 0;
}