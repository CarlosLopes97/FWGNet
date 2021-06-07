#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>

#include "ns3/applications-module.h"
#include "ns3/core-module.h"

using namespace ns3;

std::string* m_create_arr(int rows)
{
	std::string* table = new std::string[rows];
	
	for(int i = 0; i < rows; i++) 
	{
		table[i] = "loren"; 
	}
	return table;
}


std::string** m_create_mat(int rows, int columns)
{
	std::string** table = new std::string*[rows];
	
	for(int i = 0; i < rows; i++) 
	{
		table[i] = new std::string[columns]; 
		for(int j = 0; j < columns; j++)
		{ 
		table[i][j] = "loren"; 
		}// sample set value;    
	}
	return table;
}

class MyApp : public Application
{
public:

  MyApp ();
  virtual ~MyApp();


  void Setup (Ptr<Socket> socket, Address address, std::string** app_proto_ip, int n_row, int n_file_param, int n_param, int id_arrays, std::string** n_packets, std::string** arr_Times, std::string** arr_Sizes, std::string** n_rows_Size, std::string** n_rows_Time, int max_row_size, int max_row_time, std::string proto, int sum_packets, int id_proto, bool first_id);
  virtual void StartApplication (void);
  virtual void StopApplication (void);

  void ScheduleTx (void);
  void SendPacket (void);
  void Get_var(void);

  Ptr<Socket>     m_socket;
  Address         m_peer;
  EventId         m_sendEvent;
  bool            m_running;
  int             m_packetsSent;
  double          m_interval;
  int             m_size_pckts;




  int             m_n_row;
  int             m_n_param;
  int             m_id_arrays;
  int             m_sum_packets;
  int             m_id_proto;
  int             m_n_file_param;
  int             m_max_row_time;
  int             m_max_row_size;
  int             m_aux_n_Size;
  int             m_aux_n_Time;

  std::string*    m_proto;
  std::string**   m_arr_Times;
  std::string**   m_arr_Sizes;
  std::string**   m_app_proto_ip;
  std::string**   m_n_rows_Size;
  std::string**   m_n_rows_Time;
  std::string**   m_n_packets;

  std::string     aux_m_proto;
  std::string**   aux_m_arr_Times;
  std::string**   aux_m_arr_Sizes;
  std::string**   aux_m_app_proto_ip;
  std::string**   aux_m_n_rows_Size;
  std::string**   aux_m_n_rows_Time;
  std::string**   aux_m_n_packets;

  

  





  // bool            first_id;
// private:


  // std::string**   app_proto_ip;
};

MyApp::MyApp ()
  : 
    m_socket (0),
    m_peer (),
    m_sendEvent (),
    m_running (false),
    m_packetsSent (0),
    m_interval (0),
    m_size_pckts (0),

    m_n_row (0), 
    m_n_param (0),
    m_id_arrays (0),

    m_sum_packets (0),
    m_id_proto(0),
    m_n_file_param (0),
    m_max_row_time (0),
    m_max_row_size (0),
    m_aux_n_Size (0),
    m_aux_n_Time (0),

    m_proto (m_create_arr(m_n_row)),
    m_arr_Times (m_create_mat(m_max_row_time, m_n_row)),
    m_arr_Sizes (m_create_mat(m_max_row_size, m_n_row)),
    m_app_proto_ip (m_create_mat(m_n_row, m_n_param)),
    m_n_rows_Size (m_create_mat(m_n_row, m_n_file_param)),
    m_n_rows_Time (m_create_mat(m_n_row, m_n_file_param)),
    m_n_packets (m_create_mat(m_n_row, m_n_file_param)),

    aux_m_proto (""),
    aux_m_arr_Times (m_create_mat(m_max_row_time, m_n_row)),
    aux_m_arr_Sizes (m_create_mat(m_max_row_size, m_n_row)),
    aux_m_app_proto_ip (m_create_mat(m_n_row, m_n_param)),
    aux_m_n_rows_Size (m_create_mat(m_n_row, m_n_file_param)),
    aux_m_n_rows_Time (m_create_mat(m_n_row, m_n_file_param)),
    aux_m_n_packets (m_create_mat(m_n_row, m_n_file_param))

    // get_var(aux_m_proto, aux_m_arr_Times, aux_m_arr_Sizes, aux_m_app_proto_ip, aux_m_n_rows_Size, aux_m_n_rows_Time, aux_m_n_packets);
   
    
{
}

MyApp::~MyApp()
{
  m_socket = 0;
}

void
MyApp::Setup (Ptr<Socket> socket, Address address, std::string** app_proto_ip, int n_row, int n_file_param, int n_param, int id_arrays, std::string** n_packets,std::string** arr_Times, std::string** arr_Sizes, std::string** n_rows_Size, std::string** n_rows_Time, int max_row_size, int max_row_time, std::string proto, int sum_packets, int id_proto, bool first_id)
{
  m_socket = socket;
  m_peer = address;

  m_id_proto = id_proto;
  m_n_row = n_row;
  m_sum_packets = sum_packets;
  m_id_arrays = id_arrays;

  m_n_param = n_param;
  m_max_row_time = max_row_time;
  m_max_row_size = max_row_size;
  m_n_file_param = n_file_param;

  m_proto = m_create_arr(m_n_row);
  m_arr_Times = m_create_mat(m_max_row_time, m_n_row);
  m_arr_Sizes = m_create_mat(m_max_row_size, m_n_row);
  m_app_proto_ip = m_create_mat(m_n_row, m_n_param);
  m_n_rows_Size = m_create_mat(m_n_row, m_n_file_param);
  m_n_rows_Time = m_create_mat(m_n_row, m_n_file_param);
  m_n_packets = m_create_mat(m_n_row, m_n_file_param);
  
  aux_m_proto = proto;
  aux_m_arr_Times = arr_Times;
  aux_m_arr_Sizes = arr_Sizes;
  aux_m_app_proto_ip = app_proto_ip;
  aux_m_n_rows_Size = n_rows_Size;
  aux_m_n_rows_Time = n_rows_Time;
  aux_m_n_packets = n_packets;


  Get_var();

  
}

// Myapp::get_var(std::string* aux_m_proto, std::string** aux_m_arr_Times, std::string** aux_m_arr_Sizes, std::string** aux_m_app_proto_ip, std::string** aux_m_n_rows_Size, std::string** aux_m_n_rows_Time, std::string** aux_m_n_packets)
void
MyApp::Get_var (void)
{ 
  m_proto[m_id_proto] = aux_m_proto;
  m_arr_Times = aux_m_arr_Times;
  m_arr_Sizes = aux_m_arr_Sizes;

  m_app_proto_ip = aux_m_app_proto_ip;
  m_n_rows_Size = aux_m_n_rows_Size;
  m_n_rows_Time = aux_m_n_rows_Time;
  m_n_packets = aux_m_n_packets;
  m_id_proto++;
}


void
MyApp::StartApplication (void)
{
  
  // read_RV();
  std::cout<<"StartApplication"<<std::endl;
  m_running = true;
  m_packetsSent = 0;
  m_socket->Bind ();
  m_socket->Connect (m_peer);
  SendPacket ();
  
}

void
MyApp::StopApplication (void)
{
  m_running = false;

  if (m_sendEvent.IsRunning ())
    {
      std::cout<<"OK SendEvent"<<std::endl;
      Simulator::Cancel (m_sendEvent);
    }

  if (m_socket)
    {
      std::cout<<"OK IF Socket"<<std::endl;
      m_socket->Close ();
    }
}

void
MyApp::SendPacket (void)
{


  if (m_proto[m_id_arrays] == m_app_proto_ip[m_id_arrays][0])
  {
    m_aux_n_Size = stod(m_n_rows_Size[m_id_arrays][1])-1;
    m_size_pckts = std::stod(m_arr_Sizes[m_aux_n_Size][m_id_arrays]);
    m_n_rows_Size[m_id_arrays][1] = std::to_string(stod(m_n_rows_Size[m_id_arrays][1])-1); 
  }
  
  if (stod(m_n_rows_Size[m_id_arrays][1]) == 1)
  {
    m_proto[m_id_arrays] = "NULL";
    std::cout<< m_proto[m_id_arrays] <<std::endl;
    m_sum_packets = m_sum_packets - stod(m_n_packets[m_id_arrays][1]);
    // m_id_arrays++;
  }

  // Ptr<Packet> packet = Create<Packet> (m_packetSize);
  Ptr<Packet> packet = Create<Packet> (m_size_pckts);
  m_socket->Send (packet);

  if (++m_packetsSent < m_sum_packets)
    {  
      ScheduleTx ();
    }
}

void
MyApp::ScheduleTx (void)
{
  if (m_running)
    {
      if (m_proto[m_id_arrays] == m_app_proto_ip[m_id_arrays][0])
      {   
        m_aux_n_Time = stod(m_n_rows_Time[m_id_arrays][1])-1;
        m_interval = stod(m_arr_Times[m_aux_n_Time][m_id_arrays]);
        m_n_rows_Time[m_id_arrays][1] = std::to_string(stod(m_n_rows_Time[m_id_arrays][1])-1);
      }

        // std::cout<<" "<<std::endl;
        // std::cout<<"ID: "<<m_id_arrays<<std::endl;
        // std::cout<<"SUM "<<m_sum_packets<<std::endl;
        // std::cout<<"PROTO: "<<m_proto[m_id_arrays]<<std::endl;
        // std::cout<<"tNEXT "<<m_interval<<std::endl;
        // std::cout<<"APP_PROTO: "<< m_app_proto_ip[m_id_arrays][0]<<std::endl;
        // std::cout<<"SENT PACKETS "<<m_packetsSent<<std::endl;
        // std::cout<<"COUNT SENT PACKETS "<<stod(m_n_rows_Size[m_id_arrays][1])<<std::endl;
        


      Time tNext (Seconds (m_interval));
      
      if (m_id_arrays >= m_n_row-1)
      {
        m_id_arrays = 0;
      }
      else
      {
        m_id_arrays++;
      }
      // Time tNext (Seconds (m_packetSize * 8 / static_cast<double> (m_dataRate.GetBitRate ())));
      
      m_sendEvent = Simulator::Schedule (tNext, &MyApp::SendPacket, this);
      
    }
}

// void
// MyApp::m_ReadFiles (void)
// {
  
//   FILE* f = fopen("/home/carl/New_Results/Files/proto_ips.txt", "r");
  
//   // FILE* f = fopen("data.txt", "r");
//   if(f == NULL) 
//   {
//     std::cout<<"cant open file"<<std::endl;
//     std::cout<<m_proto<<" proto"<<std::endl;
  
//   }
  
//   else
//   {
//     char *get_proto;
//     char *get_ip_src;
//     char *get_ip_dst;
//     char data1[1024], data2[100], data3[100];
//     int i = 0;
//     while(fscanf(f, "%s %s %s", data1, data2, data3) == m_n_param) 
//     {
      
//       get_proto = data1;    
//       if (get_proto)  
//       m_proto[i][0] = get_proto;
      
//       get_ip_src = data2; 
//       if (get_ip_src)  
//       m_proto[i][1] = get_ip_src;
      
//       get_ip_dst = data3;
//       if (get_ip_dst)  
//       m_proto[i][2] = get_ip_dst;
//       i++;
//     }
    
//     fclose(f);
//   }
  
  
//   FILE* fS = fopen("/home/carl/New_Results/Files/list_tr_size.txt", "rb");
  
//   // FILE* f = fopen("data.txt", "r");
//   if(fS == NULL) 
//   {
//     std::cout<<"cant open file"<<std::endl;
//     std::cout<<m_proto<<" proto"<<std::endl;
  
//   }
  
//   else
//   {
    
  
    
//     char *get_proto_size;
//     char *get_s_size;
//     char data2_s[100], data1_s[1024];
//     int i = 0;
//     while((fscanf(fS, "%s %s", data1_s, data2_s)) == 2)
//     { 
//       // std::cout<<"SIZE ----> Data1: "<<data1_s<<" Data2: "<<data2_s<<std::endl;
//       // std::cout<<"OK While"<<std::endl;
//       get_proto_size = data1_s;    
//       if (get_proto_size)  
//       m_n_rows_Size[i][0] = get_proto_size;
//       m_n_packets[i][0] = get_proto_size;
      
//       get_s_size = data2_s; 
//       if (get_s_size)  
//       m_n_rows_Size[i][1] = get_s_size;
//       m_n_packets[i][1] = get_s_size;
      
//       i++;
//     }
    
//     fclose(fS);
//   }

//   FILE* fT = fopen("/home/carl/New_Results/Files/list_tr_time.txt", "r");
  
//   // FILE* f = fopen("data.txt", "r");
//   if(fT == NULL) 
//   {
//     std::cout<<"cant open file"<<std::endl;
//     std::cout<<m_proto<<" proto"<<std::endl;
  
//   }
  
//   else
//   {


    
//     char *get_proto_time;
//     char *get_s_time;
//     char data2_t[100], data1_t[1024];
//     int i = 0;
//     while(fscanf(fT, "%s %s", data1_t, data2_t) == 2) 
//     {
//       // std::cout<<"TIME ------> Data1: "<<data1_t<<" Data2: "<<data2_t<<std::endl;
//       get_proto_time = data1_t;    
//       if (get_proto_time)  
//       m_n_rows_Time[i][0] = get_proto_time;
      
//       get_s_time = data2_t; 
//       if (get_s_time)  
//       m_n_rows_Time[i][1] = get_s_time;
      
//       i++;
//     }
    
//     fclose(fT);
//   }
  
// }
// void 
// MyApp::read_RV(void)
// { 
//   m_ReadFiles();

//   for (int i = 0; i < m_n_row; ++i)
//   {
//     int aux_n_Rows_Time = stod(m_n_rows_Time[i][1]);
//     int aux_n_Rows_Size = stod(m_n_rows_Size[i][1]);  
    
//     FILE *arq_Time;
//     FILE *arq_Size;
//     char row_Time[aux_n_Rows_Time];
//     char row_Size[aux_n_Rows_Size];
//     char *aux_Time;
//     char *aux_Size;
//     char *res_Time;
//     char *res_Size;

//     m_dir_size = "/home/carl/New_Results/Files/";
//     m_dir_time = "/home/carl/New_Results/Files/";

//     m_dir_size += m_proto[i][0];
//     m_dir_size += "_size.txt";
    
//     m_dir_time += m_proto[i][0];
//     m_dir_time += "_time.txt";

//     // Abre um arquivo TEXTO para LEITURA
//     arq_Time = fopen(m_dir_time.c_str(), "rb"); 
//     arq_Size = fopen(m_dir_size.c_str(), "rb");

//     if (arq_Size == NULL)  // Se houve erro na abertura
//     {
//         std::cout<<"Unable to opem"<<std::endl;
//     }
//     if (arq_Time == NULL)  // Se houve erro na abertura
//     {
//         std::cout<<"Unable to opem"<<std::endl;
//     }
      
//       m_arr_Sizes[0][i] = m_proto[i][0];
      
//       for (int j = 1; j<aux_n_Rows_Size; ++j)
//       { 
//         // Lê uma linha (inclusive com o '\n')
//         res_Size = fgets(row_Size, aux_n_Rows_Size, arq_Size);  // Ler os caracteres ou até '\n'
//         if (res_Size)  // Se foi possível ler
//         aux_Size = row_Size;
//         m_arr_Sizes[j][i] = aux_Size;
        
//       }
      
//       m_arr_Times[0][i] = m_proto[i][0];
      
//       for (int j = 1; j<aux_n_Rows_Time; ++j)
//       { 
//         // Lê uma linha (inclusive com o '\n')
//         res_Time = fgets(row_Time, aux_n_Rows_Time, arq_Time);
//          // Ler os caracteres ou até '\n'
//         if (res_Time)  // Se foi possível ler
//         aux_Time = row_Time;
        
//         m_arr_Times[j][i] = aux_Time;
//       }

//     fclose (arq_Size);
//     fclose (arq_Time);
//   }
//   // for(int i=0;i<m_n_row;++i)
//   // {
//   //   for(int j=0;j<m_max_row_time;++j)
//   //   {
//   //     std::cout<<m_arr_Times[j][i]<<"   ";
//   //   }
//   // }

// }



