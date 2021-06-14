/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "myapp.h"

using namespace ns3;


// Default Network Topology
//
//       10.1.1.0
// n0 -------------- n1   
//    point-to-point  




NS_LOG_COMPONENT_DEFINE ("CSMAScriptExample");

// Função para geração de arrays 2d 
std::string** create_mat(int rows, int columns)
{
	std::string** table = new std::string*[rows];
	
	for(int i = 0; i < rows; i++) 
	{
		table[i] = new std::string[columns]; 
		for(int j = 0; j < columns; j++)
		{ 
		table[i][j] = ""; 
		}// sample set value;    
	}
	return table;
}
// Função para geração de arrays 1d 
std::string* create_arr(int rows)
{
	std::string* table = new std::string[rows];
	
	for(int i = 0; i < rows; i++) 
	{
		table[i] = ""; 
	}
	return table;
}

// Função para leitura de arquivos
void read_files (std::string** proto_ip, std::string* proto, std::string** n_rows_Size, std::string** n_rows_Time, std::string** n_packets_Size, std::string** n_packets_Time, int n_param ,int n_file_param, std::string case_Study)
{
  // Adicionando diretório do arquivo proto_ips.txt a uma variável
  std::string dir_f = "";
  dir_f = "/home/carl/New_Results/Files/";
  dir_f += case_Study;
  dir_f += "_proto_ips.txt";

  // FILE* f = fopen("/home/carl/New_Results/Files/"+case_Study.c_str()+"_proto_ips.txt", "r");

  FILE* f = fopen(dir_f.c_str(), "rb");
  // FILE* f = fopen("/home/carl/New_Results/Files/"+case_Study.c_str()+"_proto_ips.txt", "r");
  
  // Se encontrar falha ao abrir arquivo
  if(f == NULL)
  {
    std::cout<<"cant open file"<<std::endl;
    std::cout<<proto_ip<<" proto"<<std::endl;
  
  }
  else
  {
    // Criando varipaveis auxiliares para obter linhas do arquivo
    char *get_proto;
    char *get_ip_src;
    char *get_ip_dst;
    char data1[1024], data2[100], data3[100];
    int i = 0;
    // Scanea todo arquivo e organiza de acordo com o número de colunas
    while(fscanf(f, "%s %s %s", data1, data2, data3) == n_param) 
    {
      // Obtendo dados da coluna
      get_proto = data1;    
      // Se foi posspivel salvar dados
      if (get_proto)  
      // Salvando informações no array
      proto_ip[i][0] = get_proto;
      proto[i] = get_proto;

      // Obtendo dados da coluna
      get_ip_src = data2;
      // Se foi posspivel salvar dados
      if (get_ip_src)  
      // Salvando informações no array
      proto_ip[i][1] = get_ip_src;
      
      // Obtendo dados da coluna
      get_ip_dst = data3;
      // Se foi posspivel salvar dados
      if (get_ip_dst)  
      // Salvando informações no array
      proto_ip[i][2] = get_ip_dst;
      // Incrementa contador
      i++;
    }
    
    fclose(f);
  }

  // Adicionando diretório do arquivo list_tr_size.txt a uma variável
  std::string dir_fS = "";
  dir_fS = "/home/carl/New_Results/Files/";
  dir_fS += case_Study;
  dir_fS += "_list_tr_size.txt";
  FILE* fS = fopen(dir_fS.c_str(), "rb");

  // Se encontrar falha ao abrir arquivo
  if(fS == NULL) 
  {
    std::cout<<"cant open file"<<std::endl;
    std::cout<<proto_ip<<" proto"<<std::endl;
  
  }
  
  else
  {
    
    // Criando varipaveis auxiliares para obter linhas do arquivo
  
    
    char *get_proto_size;
    char *get_s_size;
    char data2_s[100], data1_s[1024];
    int i = 0;
    // Scanea todo arquivo e organiza de acordo com o número de colunas
    while((fscanf(fS, "%s %s", data1_s, data2_s)) == n_file_param)
    { 
      // Obtendo dados da coluna
      get_proto_size = data1_s;    
      // Se foi posspivel salvar dados
      if (get_proto_size)  
      // Salvando informações no array
      n_rows_Size[i][0] = get_proto_size;
      n_packets_Size[i][0] = n_rows_Size[i][0];
      
      // Obtendo dados da coluna
      get_s_size = data2_s; 
      if (get_s_size)  
      // Se foi posspivel salvar dados
      n_rows_Size[i][1] = get_s_size;
      // Salvando informações no array
      n_packets_Size[i][1] = n_rows_Size[i][1];

      // Incrementa contador
      i++;
      
    }
    std::cout<<"\n";
    
    fclose(fS);
  }

  // Adicionando diretório do arquivo list_tr_time.txt a uma variável
  std::string dir_fT = "";
  dir_fT = "/home/carl/New_Results/Files/";
  dir_fT += case_Study;
  dir_fT += "_list_tr_time.txt";

  FILE* fT = fopen(dir_fT.c_str(), "rb");
  
  // Se encontrar falha ao abrir arquivo
  if(fT == NULL) 
  {
    std::cout<<"cant open file"<<std::endl;
    std::cout<<proto_ip<<" proto"<<std::endl;
  
  }
  
  else
  {
    // Criando varipaveis auxiliares para obter linhas do arquivo

    
    char *get_proto_time;
    char *get_s_time;
    char data2_t[100], data1_t[1024];
    int i = 0;// Scanea todo arquivo e organiza de acordo com o número de colunas
    while(fscanf(fT, "%s %s", data1_t, data2_t) == n_file_param) 
    {
     // Obtendo dados da coluna
      get_proto_time = data1_t; // Se foi posspivel salvar dados   
      if (get_proto_time)  // Salvando informações no array
      n_rows_Time[i][0] = get_proto_time;
      n_packets_Time[i][0] = n_rows_Time[i][0];
      // Obtendo dados da coluna
      get_s_time = data2_t;// Se foi posspivel salvar dados 
      if (get_s_time) // Salvando informações no array 
      n_rows_Time[i][1] = get_s_time;
      // Salvando informações no array
      n_packets_Time[i][1] = n_rows_Time[i][1];
      std::cout<< "PROTO(TIME): "<< n_rows_Time[i][0] <<"\tSIZE(TIME): "<< n_rows_Time[i][1] <<"\tN_PCKTS(TIME): "<< n_packets_Time[i][1]<<"\n";
      // Incrementa contador
      i++;
    }
    std::cout<<"\n";
    fclose(fT);
  }

}

// Função que obtem valores de arquivos com valores gerados pelas variáveis aleatórias
void read_rv(std::string** n_rows_Size, std::string** n_rows_Time, std::string** proto_ip, std::string** arr_Sizes, std::string** arr_Times, int n_row, std::string dir_size, std::string dir_time, std::string case_Study)
{ 
  // Percorre as linhas dos arquivos
  for (int i = 0; i < n_row; ++i)
  { 
    // Definindo variáveis
    FILE *arq_Time;
    FILE *arq_Size;
    
    // Definindo diretório dos arquivos
    dir_size = "/home/carl/New_Results/Files/";
    dir_time = "/home/carl/New_Results/Files/";
    // Concatenando diretório com protocolo e parametro para chamar o arquivo
    dir_size += case_Study;
    dir_size += "_";
    dir_size += proto_ip[i][0];
    dir_size += "_size.txt";
    // Concatenando diretório com protocolo e parametro para chamar o arquivo
    dir_time += case_Study;
    dir_time += "_";
    dir_time += proto_ip[i][0];
    dir_time += "_time.txt";
    
    // Abrindo o arquivo para leitura
    arq_Time = fopen(dir_time.c_str(), "rb"); 
    arq_Size = fopen(dir_size.c_str(), "rb");
    
    // Se os arquivos não forem abertos
    if (arq_Size == NULL)  // Se houve erro na abertura
    {
        std::cout<<"Unable to open"<<std::endl;
    }
    if (arq_Time == NULL)  // Se houve erro na abertura
    {
        std::cout<<"Unable to open"<<std::endl;
    }
      
    // Adicionando nome do protocolo na linha 0
    arr_Sizes[0][i] = proto_ip[i][0];

    char *get_size;
    char data_s[1024];
    int id_size = 1;// Scanea todo arquivo e organiza de acordo com o número de colunas
    while(fscanf(arq_Size, "%s", data_s) == 1) 
    {
      // Obtendo dados da coluna
      get_size = data_s; // Se foi posspivel salvar dados   
      if (get_size)  // Salvando informações no array
      arr_Sizes[id_size][i] = get_size;    

      id_size++;
    }

    arr_Times[0][i] = proto_ip[i][0];
    char *get_time;
    char data_t[1024];
    int id_time = 1;// Scanea todo arquivo e organiza de acordo com o número de colunas
    while(fscanf(arq_Time, "%s", data_t) == 1) 
    {
      // Obtendo dados da coluna
      get_time = data_t; // Se foi posspivel salvar dados   
      if (get_time)  // Salvando informações no array
      arr_Times[id_time][i] = get_time;    

      id_time++;
    }

    // Fechando arquivos
    fclose (arq_Size);
    fclose (arq_Time);
  }
}

// Contando linhas dos arquivos
int count_Lines(std::string dir)
{
  int n_row = 0;
  std::string line;
  std::ifstream myfile(dir.c_str());
  
  if (myfile.is_open())
  {
    while (getline(myfile,line) )
    {
      n_row++;
    }
  myfile.close();
      
  }
  else std::cout <<dir << " --- Unable to open file"<<std::endl; 

  return n_row;
}

// Função princopal da simulação
int 
main (int argc, char *argv[])
{
    // Criando variáveis 
    int n_param=0;
    int n_row=0;
    int n_file_param=0;
    bool verbose;
    // Definindo strings
    std::string str_n_param = "0";
    std::string str_n_row = "0";
    std::string str_n_file_param = "0";
    std::string str_verbose = "true";

    std::string rate = "500kb/s";
    std::string lat = "2ms";
    std::string case_Study = "0";

    // Chamando variáveis do shell script
    CommandLine cmd;
    cmd.AddValue ("str_n_param", "Numero de parametros", str_n_param);
    cmd.AddValue ("str_n_row","Linhas", str_n_row);
    cmd.AddValue ("str_n_file_param","Parametros", str_n_file_param);
    cmd.AddValue ("str_verbose","verbose ativado", str_verbose);
    cmd.AddValue ("rate","Taxa", rate);
    cmd.AddValue ("lat","latencia", lat);
    cmd.AddValue ("case_Study","latencia", case_Study);
    cmd.Parse (argc, argv);

    // Convertendo string para int
    n_param = stoi(str_n_param);
    n_row = stoi(str_n_row);
    n_file_param = stoi(str_n_file_param);

    // Definindo variáveis   
    // int n_nodes = 2;
    int csma_n_nodes = 0;
    int max_row_time = 0;
    int max_row_size = 0;
    int n_lines_Time = 0;
    int n_lines_Size = 0;
    int sum_packets = 0;
    int id_arrays = 0;
    // Tempo de parada da simulação
    // int time_stop_simulation = 60.81;
    // int time_stop_simulation = 10000;

    // Criando arrays
    std::string** proto_ip = create_mat(n_row, n_param);
    std::string* proto = create_arr(n_row);
    // Definindo número de pacotes a serem enviados
    std::string** n_packets_Size = create_mat(n_row, n_file_param);
    std::string** n_packets_Time = create_mat(n_row, n_file_param);

    
    // Definindo quantidade de linhas de cada arquivo
    std::string** n_rows_Size = create_mat(n_row, n_file_param);
    std::string** n_rows_Time = create_mat(n_row, n_file_param);
    // Definindo tipo de aplicação
    std::string type = "myapp";
    std::string dir_time = "";
    std::string dir_size = "";
    // std::size_t found = 0;

    // Chamando função que lê arquivos das variáveis aleatórias
    read_files(proto_ip, proto, n_rows_Size, n_rows_Time, n_packets_Size, n_packets_Time, n_param, n_file_param, case_Study);
    

    // Percorre a quantidade de protocolos
    for(int i = 0; i<n_row; ++i)
    {
      // Definindo diretório para leitura de arquivos
      std::string dir_size = "";
      std::string dir_time = "";
      n_lines_Time = 0;
      n_lines_Size = 0; 

      dir_size = "/home/carl/New_Results/Files/";
      dir_time = "/home/carl/New_Results/Files/";


      dir_size += case_Study;
      dir_size += "_";
      dir_size += proto_ip[i][0];
      dir_size += "_size.txt";

      dir_time += case_Study;
      dir_time += "_";
      dir_time += proto_ip[i][0];
      dir_time += "_time.txt";

      // Obtendo número de linhas de cada arquivo por uma função
      n_lines_Time = count_Lines(dir_time.c_str());
      n_lines_Size = count_Lines(dir_size.c_str());
      // Somatória da quantidade de pacotes a serem enviados
      sum_packets = n_lines_Size + sum_packets;

      // Definindo o maior número de linhas dos arquivos
      if (n_lines_Time > max_row_time)
      {
          max_row_time = n_lines_Time;  
      }

      if(n_lines_Size > max_row_size)
      {
          max_row_size = n_lines_Size;
      }
    

    }

    max_row_size++;
    max_row_time++;
    // Criando array que contem os valores resultados da geração da carga de trabalho
    std::string** arr_Times = create_mat(max_row_time, n_row);
    std::string** arr_Sizes = create_mat(max_row_size, n_row);
    
    // Chamando função para atribuição de valores às variáveis
    read_rv(n_rows_Size, n_rows_Time, proto_ip, arr_Sizes, arr_Times, n_row, dir_size, dir_time, case_Study);

    // Convertendo string para boolean
    std::istringstream(str_verbose) >> verbose;

    if (verbose)
    {
        LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
    }

    // Criando nós CSMA
    NodeContainer csma_nodes;

    // Criando Container de dispositivos
    NetDeviceContainer csma_devices;

    // Definindo CSMA Helper
    CsmaHelper csma;

    // Definindo a quantidade de nós
    for(int i=0; i<n_row; ++i)
    {
        csma_n_nodes = csma_n_nodes + ceil(stoi(proto_ip[i][1]) + stoi(proto_ip[i][2]));
    }

    // Encontrar se a comunicação é por fio
    // found = proto_ip[i][0].find("eth");

    // Se encontrar a comunicação por fio, e a quantidade de nós for menor que 3
    if (csma_n_nodes > 2)
    {
      // Criaando nós
      csma_nodes.Create (csma_n_nodes);
      // Definindo atributos do CSMA
      csma.SetChannelAttribute ("DataRate", StringValue (rate));
      csma.SetChannelAttribute ("Delay", StringValue (lat));
      // Adicionando configurações aos dispositivos
      csma_devices = csma.Install (csma_nodes);
    }

    // Definindo Pilha de nós
    InternetStackHelper stack;
    stack.Install (csma_nodes);

    // Definindo ipv4 hekper
    Ipv4AddressHelper address;
    // Configurando base de endereçamento
    address.SetBase ("10.1.1.0", "255.255.255.0");
    // Adicionando configurações a um container de interface
    Ipv4InterfaceContainer csma_interfaces = address.Assign(csma_devices);
    Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
   

    // Se o tipo da aplicação for igual a "myapp"
    if (type == "myapp")
    {   
        // Create Apps
        uint16_t sinkPort = 6; // use the same for all apps

        // UDP connection from N0 to N1
        // interface of n1
        int sum_node = 0;
        int id_server = 0;
        int last_node = 0;
        std::string proto_address = "";
        
        std::string dir_ip = "";
        dir_ip = "/home/carl/New_Results/Files/";
        dir_ip += case_Study;
        dir_ip += "_ns3_ip.txt";
        
        std::ofstream myfile_ip(dir_ip);
        // std::ofstream myfile_ip("/home/carl/New_Results/Files/"+case_Study.c_str()+"_ns3_ip.txt"); 
        for(int id_proto = 0 ; id_proto < n_row; ++id_proto)
        {   
            
            sum_node = sum_node + stod(proto_ip[id_proto][1]);
            id_server = sum_node;
        
            for(int id_node = sum_node-1; id_node > last_node; --id_node)
            {            
                Address sinkAddress1 (InetSocketAddress (csma_interfaces.GetAddress (id_node), sinkPort)); 
                PacketSinkHelper packetSinkHelper1 ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort)); 
                
                //n1 as sink
                ApplicationContainer sinkApps1 = packetSinkHelper1.Install (csma_nodes.Get (id_node)); 
                
                sinkApps1.Start (Seconds (0.)); 
                // O método de parada é a quiantidade de pacotes enviados 
                // sinkApps1.Stop (Seconds (time_stop_simulation)); 
                
                proto_address = "10.1.1."+ std::to_string(id_node);

                if (myfile_ip.is_open()) 
                {
                    myfile_ip << proto[id_node] + ";" + proto_address + "\n"; 
                    myfile_ip.close(); 
                }
                last_node = sum_node;
                
                //source at n0
                Ptr<Socket> ns3UdpSocket1 = Socket::CreateSocket (csma_nodes.Get (id_server), UdpSocketFactory::GetTypeId ()); 
                // Create UDP application at n0
                Ptr<MyApp> app1 = CreateObject<MyApp> ();
                app1->Setup (ns3UdpSocket1, sinkAddress1, proto_ip, n_row, n_file_param, n_param, id_arrays, n_packets_Size, n_packets_Time, arr_Times, arr_Sizes, n_rows_Size, n_rows_Time, max_row_size, max_row_time, proto[0], sum_packets, id_proto, true);
                csma_nodes.Get (id_server)->AddApplication (app1);
                app1->SetStartTime (Seconds (0.));
                // O método de parada é a quiantidade de pacotes enviados
                // app1->SetStopTime (Seconds (time_stop_simulation));
            }
        }
    }
    // Criando arquivo de trace
    csma.EnablePcapAll (""+case_Study+"_test");


    Simulator::Run ();
    Simulator::Destroy ();
    return 0;
}



