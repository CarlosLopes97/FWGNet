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
void read_files (std::string** proto_ip, std::string* proto, std::string** n_rows_Size, std::string** n_rows_Time, std::string** n_packets, int n_param ,int n_file_param, std::string case_Study)
{
  // Adicionando diretório do arquivo proto_ips.txt a uma variável
  FILE* f = fopen("/home/carl/New_Results/Files/"+case_Study.c_str()+"_proto_ips.txt", "r");
  
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
  FILE* fS = fopen("/home/carl/New_Results/Files/"+case_Study.c_str()++"_list_tr_size.txt", "rb");
  
  // FILE* f = fopen("data.txt", "r");
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
      n_packets[i][0] = get_proto_size;
      
      // Obtendo dados da coluna
      get_s_size = data2_s; 
      if (get_s_size)  
      // Se foi posspivel salvar dados
      n_rows_Size[i][1] = get_s_size;
      // Salvando informações no array
      n_packets[i][1] = get_s_size;
      
      // Incrementa contador
      i++;
    }
    
    fclose(fS);
  }

  // Adicionando diretório do arquivo list_tr_time.txt a uma variável
  FILE* fT = fopen("/home/carl/New_Results/Files/"+case_Study.c_str()++"_list_tr_time.txt", "r");
  
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
      // Obtendo dados da coluna
      get_s_time = data2_t;// Se foi posspivel salvar dados 
      if (get_s_time) // Salvando informações no array 
      n_rows_Time[i][1] = get_s_time;
      // Incrementa contador
      i++;
    }
    
    fclose(fT);
  }

  
}
// Função que obtem valores de arquivos com valores gerados pelas variáveis aleatórias
void read_rv(std::string** n_rows_Size, std::string** n_rows_Time, std::string** proto_ip, std::string** arr_Sizes, std::string** arr_Times, int n_row, std::string dir_size, std::string dir_time)
{ 
  // Percorre as linhas dos arquivos
  for (int i = 0; i < n_row; ++i)
  {
    // Definindo tamanho da variável auxiliar
    int aux_n_Rows_Time = stoi(n_rows_Time[i][1]);
    int aux_n_Rows_Size = stoi(n_rows_Size[i][1]);  
    
    // Definindo variáveis
    FILE *arq_Time;
    FILE *arq_Size;
    char row_Time[aux_n_Rows_Time];
    char row_Size[aux_n_Rows_Size];
    char *aux_Time;
    char *aux_Size;
    char *res_Time;
    char *res_Size;
    
    // Definindo diretório dos arquivos
    dir_size = "/home/carl/New_Results/Files/";
    dir_time = "/home/carl/New_Results/Files/";
    // Concatenando diretório com protocolo e parametro para chamar o arquivo
    dir_size += proto_ip[i][0];
    dir_size += "_size.txt";
    // Concatenando diretório com protocolo e parametro para chamar o arquivo
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
      
      // Percorrendo todas as linhas dos arquivos
      for (int j = 1; j<aux_n_Rows_Size; ++j)
      { 
        // Lê uma linha (inclusive com o '\n')
        res_Size = fgets(row_Size, aux_n_Rows_Size, arq_Size);  
        // Se foi possível ler
        if (res_Size)  
        aux_Size = row_Size;
        // Adicionando valor da linha no array
        arr_Sizes[j][i] = aux_Size;
      }
      
      arr_Times[0][i] = proto_ip[i][0];
      
      for (int j = 1; j<aux_n_Rows_Time; ++j)
      { 
        // Lê uma linha (inclusive com o '\n')
        res_Time = fgets(row_Time, aux_n_Rows_Time, arq_Time);
        // Se foi possível ler
        if (res_Time)  
        aux_Time = row_Time;
        // Adicionando valor da linha no array
        arr_Times[j][i] = aux_Time;
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
  else std::cout <<dir << "Unable to open file"<<std::endl; 

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
  int n_nodes = 2;
  int p2p_n_nodes = 0;
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
  std::string** n_packets = create_mat(n_row, n_file_param);
  // Definindo quantidade de linhas de cada arquivo
  std::string** n_rows_Size = create_mat(n_row, n_file_param);
  std::string** n_rows_Time = create_mat(n_row, n_file_param);
  // Definindo tipo de aplicação
  std::string type = "myapp";
  std::string dir_time = "";
  std::string dir_size = "";
  std::size_t found = 0;
  
  // Chamando função que lê arquivos das variáveis aleatórias
  read_files(proto_ip, proto, n_rows_Size, n_rows_Time, n_packets, n_param, n_file_param, case_Study);
  

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
  
    dir_size += proto_ip[i][0];
    dir_size += "_size.txt";
    
    
    dir_time += proto_ip[i][0];
    dir_time += "_time.txt";

    // Obtendo número de linhas de cada arquivo por uma função
    n_lines_Time = count_Lines(dir_time);
    n_lines_Size = count_Lines(dir_size);
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
  std::string** arr_Sizes = create_mat(max_row_time, n_row);


  // Chamando função para atribuição de valores às variáveis
  read_rv(n_rows_Size, n_rows_Time, proto_ip, arr_Sizes, arr_Times, n_row, dir_size, dir_time);
  
  // Convertendo string para boolean
  std::istringstream(str_verbose) >> verbose;

  if (verbose)
    {
      LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
      LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);
    }

  // Criando nós P2P
  NodeContainer p2p_nodes;

  // Criando Container de dispositivos
  NetDeviceContainer p2p_devices;

  // Definindo P2P Helper
  PointToPointHelper p2p;
  
  // Definindo a quantidade de nós
  p2p_n_nodes = ceil(stoi(proto_ip[0][1]) + stoi(proto_ip[0][2]));
  
  // Encontrar se a comunicação é por fio
  found = proto_ip[0][0].find("eth");
  
  // Se encontrar a comunicação por fio, e a quantidade de nós for menor que 3
  if (found!=std::string::npos && n_nodes < 3)
  {
    // Criaando nós
    p2p_nodes.Create (p2p_n_nodes);
    // Definindo atributos do P2P
    p2p.SetDeviceAttribute ("DataRate", StringValue (rate));
    p2p.SetChannelAttribute ("Delay", StringValue (lat));
    // Adicionando configurações aos dispositivos
    p2p_devices = p2p.Install (p2p_nodes);
  }
  
  // Definindo Pilha de nós
  InternetStackHelper stack;
  stack.Install (p2p_nodes);
  
  // Definindo ipv4 hekper
  Ipv4AddressHelper address;
  // Configurando base de endereçamento
  address.SetBase ("10.1.1.0", "255.255.255.0");
  // Adicionando configurações a um container de interface
  Ipv4InterfaceContainer p2p_interfaces = address.Assign(p2p_devices);
  
  // Se o tipo da aplicação for igual a "udpClientEcho"
  if (type == "udpclientecho")
  {
    UdpEchoServerHelper echoServer (9);

    ApplicationContainer serverApps = echoServer.Install (p2p_nodes.Get(0));
    serverApps.Start (Seconds (1.0));
    serverApps.Stop (Seconds (100.0));

    UdpEchoClientHelper echoClient (p2p_interfaces.GetAddress (1), 9);
    echoClient.SetAttribute ("MaxPackets", UintegerValue (100));
    echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
    echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

    ApplicationContainer clientApps = echoClient.Install (p2p_nodes.Get (1));
    clientApps.Start (Seconds (2.0));
    clientApps.Stop (Seconds (100.0));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
  }
  
  // Se o tipo da aplicação for igual a "myapp"
  if (type == "myapp")
  {   
    // Create Apps
    uint16_t sinkPort = 6; // use the same for all apps
    
    // UDP connection from N0 to N1
    // interface of n1
    Address sinkAddress1 (InetSocketAddress (p2p_interfaces.GetAddress (1), sinkPort)); 
    PacketSinkHelper packetSinkHelper1 ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
    //n1 as sink
    ApplicationContainer sinkApps1 = packetSinkHelper1.Install (p2p_nodes.Get (1)); 
    sinkApps1.Start (Seconds (0.));
    // O método de parada é a quiantidade de pacotes enviados
    // sinkApps1.Stop (Seconds (time_stop_simulation));

    std::ofstream myfile_ip_1("/home/carl/New_Results/Files/"+case_Study.c_str()++"_ns3_ip.txt");
    if (myfile_ip_1.is_open()){
      myfile_ip_1 << proto[0] + ";" + "10.1.1."+std::to_string(1) + "\n";
      myfile_ip_1 << proto[1] + ";" + "10.1.1."+std::to_string(2) + "\n";
      myfile_ip_1.close();
    }

    //source at n0
    Ptr<Socket> ns3UdpSocket1 = Socket::CreateSocket (p2p_nodes.Get (0), UdpSocketFactory::GetTypeId ()); 
    
    // Create UDP application at n0
    Ptr<MyApp> app1 = CreateObject<MyApp> ();
    app1->Setup (ns3UdpSocket1, sinkAddress1, proto_ip, n_row, n_file_param, n_param, id_arrays, n_packets, arr_Times, arr_Sizes, n_rows_Size, n_rows_Time, max_row_size, max_row_time, proto[0], sum_packets, 0, true);
    p2p_nodes.Get (0)->AddApplication (app1);
    app1->SetStartTime (Seconds (0.));
    // O método de parada é a quiantidade de pacotes enviados
    // app1->SetStopTime (Seconds (time_stop_simulation));



    Address sinkAddress2 (InetSocketAddress (p2p_interfaces.GetAddress (0), sinkPort)); // interface of n0
    PacketSinkHelper packetSinkHelper2 ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
    ApplicationContainer sinkApps2 = packetSinkHelper2.Install (p2p_nodes.Get (0)); //n2 as sink
    sinkApps2.Start (Seconds (0.));
    // O método de parada é a quiantidade de pacotes enviados
    // sinkApps2.Stop (Seconds (time_stop_simulation));
    
    //source at n1
    Ptr<Socket> ns3UdpSocket2 = Socket::CreateSocket (p2p_nodes.Get (1), UdpSocketFactory::GetTypeId ()); 

    // Create UDP application at n1
    Ptr<MyApp> app2 = CreateObject<MyApp> ();
    app2->Setup (ns3UdpSocket2, sinkAddress2, proto_ip, n_row, n_file_param, n_param, id_arrays, n_packets, arr_Times, arr_Sizes, n_rows_Size, n_rows_Time, max_row_size, max_row_time, proto[1], sum_packets, 1, false);
    p2p_nodes.Get (1)->AddApplication (app2);
    app2->SetStartTime (Seconds (0.));
    // O método de parada é a quiantidade de pacotes enviados
    // app2->SetStopTime (Seconds (time_stop_simulation));
    
  }
  // Criando arquivo de trace
  p2p.EnablePcapAll (""+case_Study+"");


  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}



