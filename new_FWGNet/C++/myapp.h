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


  void Setup (Ptr<Socket> socket, Address address, std::string** app_proto_ip, int n_row, int n_file_param, int n_param, int id_arrays, std::string** n_packets_Size, std::string** n_packets_Time, std::string** arr_Times, std::string** arr_Sizes, std::string** n_rows_Size, std::string** n_rows_Time, int max_row_size, int max_row_time, std::string proto, int sum_packets, int id_proto, bool first_id);
  virtual void StartApplication (void);
  virtual void StopApplication (void);

  void ScheduleTx (void);
  void SendPacket (void);
  void Get_var(void);
  int             count_apps;
  Ptr<Socket>     m_socket;
  Address         m_peer;
  EventId         m_sendEvent;
  bool            m_running;
  int             m_packetsSent;
  int             m_count;
  double          m_interval;
  int             m_size_pckts;
  double          time_s;




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
  std::string**   m_n_packets_Size;
  std::string**   m_n_packets_Time;

  std::string     aux_m_proto;
  std::string**   aux_m_arr_Times;
  std::string**   aux_m_arr_Sizes;
  std::string**   aux_m_app_proto_ip;
  std::string**   aux_m_n_rows_Size;
  std::string**   aux_m_n_rows_Time;
  std::string**   aux_m_n_packets_Size;
  std::string**   aux_m_n_packets_Time;

};

MyApp::MyApp ()
  : 
    count_apps (0),
    m_socket (0),
    m_peer (),
    m_sendEvent (),
    m_running (false),
    m_packetsSent (0),
    m_count (1),
    m_interval (0),
    m_size_pckts (0),
    time_s (0),

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
    m_n_packets_Size (m_create_mat(m_n_row, m_n_file_param)),
    m_n_packets_Time (m_create_mat(m_n_row, m_n_file_param)),

    aux_m_proto (""),
    aux_m_arr_Times (m_create_mat(m_max_row_time, m_n_row)),
    aux_m_arr_Sizes (m_create_mat(m_max_row_size, m_n_row)),
    aux_m_app_proto_ip (m_create_mat(m_n_row, m_n_param)),
    aux_m_n_rows_Size (m_create_mat(m_n_row, m_n_file_param)),
    aux_m_n_rows_Time (m_create_mat(m_n_row, m_n_file_param)),
    aux_m_n_packets_Size (m_create_mat(m_n_row, m_n_file_param)),
    aux_m_n_packets_Time (m_create_mat(m_n_row, m_n_file_param))


    // get_var(aux_m_proto, aux_m_arr_Times, aux_m_arr_Sizes, aux_m_app_proto_ip, aux_m_n_rows_Size, aux_m_n_rows_Time, aux_m_n_packets);
   
    
{
}

MyApp::~MyApp()
{
  m_socket = 0;
}

void
MyApp::Setup (Ptr<Socket> socket, Address address, std::string** app_proto_ip, int n_row, int n_file_param, int n_param, int id_arrays, std::string** n_packets_Size, std::string** n_packets_Time, std::string** arr_Times, std::string** arr_Sizes, std::string** n_rows_Size, std::string** n_rows_Time, int max_row_size, int max_row_time, std::string proto, int sum_packets, int id_proto, bool first_id)
{
  // count_apps=count_apps+1;
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
  m_n_packets_Size = m_create_mat(m_n_row, m_n_file_param);
  m_n_packets_Time = m_create_mat(m_n_row, m_n_file_param);
  
  aux_m_proto = proto;
  aux_m_arr_Times = arr_Times;
  aux_m_arr_Sizes = arr_Sizes;
  aux_m_app_proto_ip = app_proto_ip;
  aux_m_n_rows_Size = n_rows_Size;
  aux_m_n_rows_Time = n_rows_Time;
  aux_m_n_packets_Size = m_n_packets_Size;
  aux_m_n_packets_Time = m_n_packets_Time;
  
  // std::cout<<"0"<<std::endl;
  std::cout<<"SETUP COUNT: "<<m_id_proto<<std::endl;
  std::cout<<"SETUP N_ROW: "<<m_n_row<<std::endl;
  // std::cout<<"SETUP APP_PROTO: "<< aux_m_app_proto_ip<<std::endl;


  // Get_var();

  
}

// Myapp::get_var(std::string* aux_m_proto, std::string** aux_m_arr_Times, std::string** aux_m_arr_Sizes, std::string** aux_m_app_proto_ip, std::string** aux_m_n_rows_Size, std::string** aux_m_n_rows_Time, std::string** aux_m_n_packets)
void
MyApp::Get_var (void)
{ 
  // m_proto[m_id_proto] = aux_m_proto;
  m_arr_Times = aux_m_arr_Times;
  m_arr_Sizes = aux_m_arr_Sizes;

  m_app_proto_ip = aux_m_app_proto_ip;
  m_n_rows_Size = aux_m_n_rows_Size;
  m_n_rows_Time = aux_m_n_rows_Time;
  // m_n_packets = aux_m_n_packets;
  // std::cout<<"Get var ID: "<<m_id_proto<<std::endl;
  // std::cout<<"Get var PROTO: "<<m_proto[m_id_proto]<<std::endl;
  // std::cout<<"Get var APP_PROTO: "<< m_app_proto_ip[m_id_proto][0]<<std::endl;
  // m_id_proto++;
  
}


void
MyApp::StartApplication (void)
{
  // if(m_sum_packets > m_packetsSent)
  if(m_id_proto == m_n_row-1)
  {
    Get_var();
    // std::cout<<"1"<<std::endl;
    m_running = true;
    // m_packetsSent = 0;
    m_socket->Bind ();
    m_socket->Connect (m_peer);
    SendPacket ();
  }
  // else
  // {
  //   StopApplication ();
  // }

}


void
MyApp::StopApplication (void)
{
  // std::cout<<"2"<<std::endl;
  m_running = false;

  if (m_sendEvent.IsRunning ())
    {
      Simulator::Cancel (m_sendEvent);
    }

  if (m_socket)
    {
      m_socket->Close ();
    }
}

void
MyApp::SendPacket (void)
{  
  // Se o número de linhas for maior que 0
  if (stod(m_n_rows_Size[m_id_arrays][1]) > 0)
  {
    // Variável auxiliar recebe o numero de linhas
    m_aux_n_Size = stod(m_n_rows_Size[m_id_arrays][1]);

    // O tamanho do pacote é definido
    m_size_pckts = stod(m_arr_Sizes[m_aux_n_Size][m_id_arrays]);

    // O número da linha decrementa 1
    m_n_rows_Size[m_id_arrays][1] = std::to_string(stod(m_n_rows_Size[m_id_arrays][1])-1); 
    
    // Cria-se um pacote com o tamanho definido
    Ptr<Packet> packet = Create<Packet> (m_size_pckts);
    
    // Envia o pacote para o socket
    m_socket->Send (packet);
    
    m_packetsSent++;
    
    // Se a somatória é maior que a quantidade de pacotes enviados então segue-se para Schedule
    if (m_packetsSent < m_sum_packets)
    {    
      ScheduleTx ();
    }else{
      // Finaliza a aplicação
      StopApplication ();
    }

  }else
  {
    // Incrementa e decrementa o id dos arrays
    if (m_id_arrays >= m_n_row-1)
    {
      m_id_arrays = 0;
    }else
    {
      m_id_arrays++;
    }
    // Retorna à função de SendPackets
    SendPacket();
  }
}

void
MyApp::ScheduleTx (void)
{
  // Se a aplicação está executando
  if (m_running)
    {    
      // Se o número de linhas for maior que 0
      if (stod(m_n_rows_Time[m_id_arrays][1]) > 0)
      {
        // Variável auxiliar recebe o numero de linhas
        m_aux_n_Time = stod(m_n_rows_Time[m_id_arrays][1]);

        // O intervalo é definido
        m_interval = stod(m_arr_Times[m_aux_n_Time][m_id_arrays]);
        
        // O número da linha decrementa 1
        m_n_rows_Time[m_id_arrays][1] = std::to_string(stod(m_n_rows_Time[m_id_arrays][1])-1);

        // Defini-se o tNext
        Time tNext (Seconds (m_interval));

        // Incrementa e decrementa o id dos arrays
        if (m_id_arrays >= m_n_row-1)
            {
              m_id_arrays = 0;
            }else
            {
              m_id_arrays++;
            }
            
        // Envia um evento
        m_sendEvent = Simulator::Schedule (tNext, &MyApp::SendPacket, this);
      }else{
         // Incrementa e decrementa o id dos arrays
         if (m_id_arrays >= m_n_row-1)
          {
            m_id_arrays = 0;
          }else
          {
            m_id_arrays++;
          }
        // Retorna à função de SendPackets
        SendPacket();
      }
   
    }
}