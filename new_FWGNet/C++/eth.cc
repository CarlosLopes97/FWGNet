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
// #include "ns3/wifi-module.h"
using namespace ns3;


// Default Network Topology
//
//       10.1.1.0
// n0 -------------- n1   
//    point-to-point  




NS_LOG_COMPONENT_DEFINE ("SecondScriptExample");

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

std::string* create_arr(int rows)
{
	std::string* table = new std::string[rows];
	
	for(int i = 0; i < rows; i++) 
	{
		table[i] = ""; 
	}
	return table;
}

void ReadFiles (std::string** proto_ip, std::string* proto, std::string** n_rows_Size, std::string** n_rows_Time, std::string** n_packets, int n_param ,int n_file_param)
{
  
  FILE* f = fopen("/home/carl/New_Results/Files/proto_ips.txt", "r");
  
  // FILE* f = fopen("data.txt", "r");
  if(f == NULL)
  {
    std::cout<<"cant open file"<<std::endl;
    std::cout<<proto_ip<<" proto"<<std::endl;
  
  }
  else
  {
    char *get_proto;
    char *get_ip_src;
    char *get_ip_dst;
    char data1[1024], data2[100], data3[100];
    int i = 0;
    while(fscanf(f, "%s %s %s", data1, data2, data3) == n_param) 
    {
      
      get_proto = data1;    
      if (get_proto)  
      proto_ip[i][0] = get_proto;
      proto[i] = get_proto;
      get_ip_src = data2; 
      if (get_ip_src)  
      proto_ip[i][1] = get_ip_src;
      
      get_ip_dst = data3;
      if (get_ip_dst)  
      proto_ip[i][2] = get_ip_dst;
      i++;
    }
    
    fclose(f);
  }

  
  FILE* fS = fopen("/home/carl/New_Results/Files/list_tr_size.txt", "rb");
  
  // FILE* f = fopen("data.txt", "r");
  if(fS == NULL) 
  {
    std::cout<<"cant open file"<<std::endl;
    std::cout<<proto_ip<<" proto"<<std::endl;
  
  }
  
  else
  {
    
  
    
    char *get_proto_size;
    char *get_s_size;
    char data2_s[100], data1_s[1024];
    int i = 0;
    while((fscanf(fS, "%s %s", data1_s, data2_s)) == n_file_param)
    { 
     
      // std::cout<<"SIZE ----> Data1: "<<data1_s<<" Data2: "<<data2_s<<std::endl;
      // std::cout<<"OK While"<<std::endl;
      get_proto_size = data1_s;    
      if (get_proto_size)  
      n_rows_Size[i][0] = get_proto_size;
      n_packets[i][0] = get_proto_size;
      get_s_size = data2_s; 
      
      if (get_s_size)  
      n_rows_Size[i][1] = get_s_size;
      n_packets[i][1] = get_s_size;
     
      i++;
    }
    
    fclose(fS);
  }


  FILE* fT = fopen("/home/carl/New_Results/Files/list_tr_time.txt", "r");
  
  // FILE* f = fopen("data.txt", "r");
  if(fT == NULL) 
  {
    std::cout<<"cant open file"<<std::endl;
    std::cout<<proto_ip<<" proto"<<std::endl;
  
  }
  
  else
  {


    
    char *get_proto_time;
    char *get_s_time;
    char data2_t[100], data1_t[1024];
    int i = 0;
    while(fscanf(fT, "%s %s", data1_t, data2_t) == n_file_param) 
    {
      // std::cout<<"TIME ------> Data1: "<<data1_t<<" Data2: "<<data2_t<<std::endl;
      get_proto_time = data1_t;    
      if (get_proto_time)  
      n_rows_Time[i][0] = get_proto_time;
      
      get_s_time = data2_t; 
      if (get_s_time)  
      n_rows_Time[i][1] = get_s_time;
      
      i++;
    }
    
    fclose(fT);
  }

  
}

void read_RV(std::string** n_rows_Size, std::string** n_rows_Time, std::string** proto_ip, std::string** arr_Sizes, std::string** arr_Times, int n_row, std::string dir_size, std::string dir_time)
{ 

  for (int i = 0; i < n_row; ++i)
  {
    int aux_n_Rows_Time = stoi(n_rows_Time[i][1]);

    int aux_n_Rows_Size = stoi(n_rows_Size[i][1]);  

    FILE *arq_Time;
    FILE *arq_Size;
    char row_Time[aux_n_Rows_Time];
    char row_Size[aux_n_Rows_Size];
    char *aux_Time;
    char *aux_Size;
    char *res_Time;
    char *res_Size;

    dir_size = "/home/carl/New_Results/Files/";
    dir_time = "/home/carl/New_Results/Files/";

    dir_size += proto_ip[i][0];
    dir_size += "_size.txt";
    
    dir_time += proto_ip[i][0];
    dir_time += "_time.txt";
    

    // Abre um arquivo TEXTO para LEITURA
    arq_Time = fopen(dir_time.c_str(), "rb"); 
    arq_Size = fopen(dir_size.c_str(), "rb");
    
    if (arq_Size == NULL)  // Se houve erro na abertura
    {
        std::cout<<"Unable to opem"<<std::endl;
    }
    if (arq_Time == NULL)  // Se houve erro na abertura
    {
        std::cout<<"Unable to opem"<<std::endl;
    }
      
      arr_Sizes[0][i] = proto_ip[i][0];
      
      for (int j = 1; j<aux_n_Rows_Size; ++j)
      { 
        // Lê uma linha (inclusive com o '\n')
        res_Size = fgets(row_Size, aux_n_Rows_Size, arq_Size);  // Ler os caracteres ou até '\n'
        if (res_Size)  // Se foi possível ler
        aux_Size = row_Size;
        arr_Sizes[j][i] = aux_Size;
      }
      
      arr_Times[0][i] = proto_ip[i][0];
      
      for (int j = 1; j<aux_n_Rows_Time; ++j)
      { 
        // Lê uma linha (inclusive com o '\n')
        res_Time = fgets(row_Time, aux_n_Rows_Time, arq_Time);
         // Ler os caracteres ou até '\n'
        if (res_Time)  // Se foi possível ler
        aux_Time = row_Time;
        
        arr_Times[j][i] = aux_Time;
      }

    fclose (arq_Size);
    fclose (arq_Time);
  }
}

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
  else std::cout <<dir << " ------ Unable to open file"<<std::endl; 

  return n_row;
}


int 
main (int argc, char *argv[])
{

  int n_param=0;
  int n_row=0;
  int n_file_param=0;
  bool verbose;

  std::string str_n_param = "0";
  std::string str_n_row = "0";
  std::string str_n_file_param = "0";
  std::string str_verbose = "true";
  
  std::string rate = "500kb/s";
  std::string lat = "2ms";

  CommandLine cmd;
  cmd.AddValue ("str_n_param", "Numero de parametros", str_n_param);
  cmd.AddValue ("str_n_row","Linhas", str_n_row);
  cmd.AddValue ("str_n_file_param","Parametros", str_n_file_param);
  cmd.AddValue ("str_verbose","verbose ativado", str_verbose);
  cmd.AddValue ("rate","Taxa", rate);
  cmd.AddValue ("lat","latencia", lat);
  cmd.Parse (argc, argv);

  n_param = stoi(str_n_param);
  n_row = stoi(str_n_row);
  n_file_param = stoi(str_n_file_param);
    
  int n_nodes = 2;
  int p2p_n_nodes = 0;
  int max_row_time = 0;
  int max_row_size = 0;
  int n_lines_Time = 0;
  int n_lines_Size = 0;
  int sum_packets = 0;
  int id_arrays = 0;
  // int time_stop_simulation = 60.81;
  // int time_stop_simulation = 10000;

  std::string** proto_ip = create_mat(n_row, n_param);
  std::string* proto = create_arr(n_row);
  std::string** n_packets = create_mat(n_row, n_file_param);
  std::string** n_rows_Size = create_mat(n_row, n_file_param);
  std::string** n_rows_Time = create_mat(n_row, n_file_param);
  std::string type = "myapp";
  std::string dir_time = "";
  std::string dir_size = "";
  std::size_t found = 0;
  // cmd.Parse (argc,argv);
 
  ReadFiles(proto_ip, proto, n_rows_Size, n_rows_Time, n_packets, n_param, n_file_param);
  


 for(int i = 0; i<n_row; ++i)
  {
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

    n_lines_Time = count_Lines(dir_time);
    n_lines_Size = count_Lines(dir_size);

    sum_packets = n_lines_Size + sum_packets;
    
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
  std::string** arr_Times = create_mat(max_row_time, n_row);
  std::string** arr_Sizes = create_mat(max_row_time, n_row);



  read_RV(n_rows_Size, n_rows_Time, proto_ip, arr_Sizes, arr_Times, n_row, dir_size, dir_time);
  
  // verbose = boost::lexical_cast<bool>(str_verbose);
  std::istringstream(str_verbose) >> verbose;
  // uint32_t nCsma = 3;

  if (verbose)
    {
      LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
      LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);
    }

  NodeContainer p2p_nodes;
  NetDeviceContainer p2p_devices;

  // NodeContainer csma_nodes;
  // NetDeviceContainer csma_devices;

  // NodeContainer wifi_nodes;
  // NetDeviceContainer wifi_devices;

  PointToPointHelper p2p;
  // CsmaHelper csma;
  // WifiHelper wifi;
  // WifiMacHelper wifiMac;
  

  p2p_n_nodes = ceil(stoi(proto_ip[0][1]) + stoi(proto_ip[0][2]));

  found = proto_ip[0][0].find("eth");

  if (found!=std::string::npos && n_nodes < 3)
  {
    
   

    p2p_nodes.Create (p2p_n_nodes);
    p2p.SetDeviceAttribute ("DataRate", StringValue (rate));
    p2p.SetChannelAttribute ("Delay", StringValue (lat));
    p2p_devices = p2p.Install (p2p_nodes);
  }
    // if (found!=std::string::npos && n_nodes > 2)
    // {	
    //   csma_n_nodes = n_nodes;
    //   // csmaNodes.Add (p2p_nodes.Get (1));
    //   csma_nodes.Create (csma_n_nodes);
    //   csma.SetChannelAttribute ("DataRate", DataRateValue (DataRate (rate)));
    //   csma.SetChannelAttribute ("Delay", StringValue (lat));
    //   csma_devices = csma.Install (csma_nodes);
    // }

    // found = proto_ip[i][0].find("radio");
    // if (found!=std::string::npos)
    // {	
    //   wifi_n_nodes = n_nodes;
    //   wifi_nodes.Create (wifi_n_nodes);
    //   YansWifiPhyHelper wifiPhy =  YansWifiPhyHelper::Default ();
    //   wifi.SetStandard(WIFI_PHY_STANDARD_80211b);
    //   YansWifiChannelHelper wifiChannel;
    //   wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
    //   wifiChannel.AddPropagationLoss ("ns3::FriisPropagationLossModel");
    //   wifiPhy.SetChannel (wifiChannel.Create ());
    //   std::string phyMode ("DsssRate2Mbps");
    //   wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
    //                 "DataMode",StringValue (phyMode),
    //                 "ControlMode",StringValue (phyMode));
    //   wifiMac.SetType ("ns3::AdhocWifiMac");
    //   wifi_devices = wifi.Install (wifiPhy, wifiMac, wifi_nodes);
    // }	

  

  InternetStackHelper stack;
  stack.Install (p2p_nodes);
  // stack.Install (csma_nodes);

  
  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer p2p_interfaces = address.Assign(p2p_devices);
      
  // std::string** n_rows_Size = create_mat(n_row, 2);
  // std::string** n_rows_Time = create_mat(n_row, 2);



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
  
  if (type == "myapp")
  {   
    
    // Create Apps
    uint16_t sinkPort = 6; // use the same for all apps
    // UDP connection from N0 to N24

    Address sinkAddress1 (InetSocketAddress (p2p_interfaces.GetAddress (1), sinkPort)); // interface of n24
    PacketSinkHelper packetSinkHelper1 ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
    ApplicationContainer sinkApps1 = packetSinkHelper1.Install (p2p_nodes.Get (1)); //n2 as sink
    sinkApps1.Start (Seconds (0.));
    // sinkApps1.Stop (Seconds (time_stop_simulation));

    std::ofstream myfile_ip_1("/home/carl/New_Results/Files/ns3_ip.txt");
    if (myfile_ip_1.is_open()){
      myfile_ip_1 << proto[0] + ";" + "10.1.1."+std::to_string(1) + "\n";
      myfile_ip_1 << proto[1] + ";" + "10.1.1."+std::to_string(2) + "\n";
      myfile_ip_1.close();
    }


    Ptr<Socket> ns3UdpSocket1 = Socket::CreateSocket (p2p_nodes.Get (0), UdpSocketFactory::GetTypeId ()); //source at n0
    
    // Create UDP application at n0
    // std::string app = "app1";
    Ptr<MyApp> app1 = CreateObject<MyApp> ();
    
    app1->Setup (ns3UdpSocket1, sinkAddress1, proto_ip, n_row, n_file_param, n_param, id_arrays, n_packets, arr_Times, arr_Sizes, n_rows_Size, n_rows_Time, max_row_size, max_row_time, proto[0], sum_packets, 0, true);
    
    p2p_nodes.Get (0)->AddApplication (app1);

    app1->SetStartTime (Seconds (0.));
    // app1->SetStopTime (Seconds (time_stop_simulation));



    Address sinkAddress2 (InetSocketAddress (p2p_interfaces.GetAddress (0), sinkPort)); // interface of n24
    PacketSinkHelper packetSinkHelper2 ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
    ApplicationContainer sinkApps2 = packetSinkHelper2.Install (p2p_nodes.Get (0)); //n2 as sink
    sinkApps2.Start (Seconds (0.));
    // sinkApps2.Stop (Seconds (time_stop_simulation));
    
    
    Ptr<Socket> ns3UdpSocket2 = Socket::CreateSocket (p2p_nodes.Get (1), UdpSocketFactory::GetTypeId ()); //source at n0

    // Create UDP application at n0
    // std::string app = "app2";
    Ptr<MyApp> app2 = CreateObject<MyApp> ();
    app2->Setup (ns3UdpSocket2, sinkAddress2, proto_ip, n_row, n_file_param, n_param, id_arrays, n_packets, arr_Times, arr_Sizes, n_rows_Size, n_rows_Time, max_row_size, max_row_time, proto[1], sum_packets, 1, false);
    p2p_nodes.Get (1)->AddApplication (app2);
    app2->SetStartTime (Seconds (0.));
    // app2->SetStopTime (Seconds (time_stop_simulation));
    
    
    // Address sinkAddress2 (InetSocketAddress (p2p_interfaces.GetAddress (0), sinkPort)); // interface of n24
    // PacketSinkHelper packetSinkHelper2 ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), sinkPort));
    // ApplicationContainer sinkApps2 = packetSinkHelper2.Install (p2p_nodes.Get (0)); //n2 as sink
    // sinkApps2.Start (Seconds (0.));
    // sinkApps2.Stop (Seconds (1000));

    // Ptr<Socket> ns3UdpSocket2 = Socket::CreateSocket (p2p_nodes.Get (1), UdpSocketFactory::GetTypeId ()); //source at n0

    // // Create UDP application at n0
    // // std::string app = "app2";
    // Ptr<MyApp> app2 = CreateObject<MyApp> ();
    // app2->Setup (ns3UdpSocket2, sinkAddress2, packetSize, n_packets, DataRate ("2Mbps"), proto_ip, n_row, n_file_param, n_param, max_row_time, max_row_size, interval, size_pckts, id_arrays, arr_Times, arr_Sizes, n_rows_Size, n_rows_Time, dir_size, dir_time, aux_n_Size, aux_n_Time, proto_ip, n_file_param, sum_packets);
    // p2p_nodes.Get (0)->AddApplication (app2);
    // app2->SetStartTime (Seconds (0.));
    // app2->SetStopTime (Seconds (1000));
  }
  p2p.EnablePcapAll ("huehuehue");


  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}



