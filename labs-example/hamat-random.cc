/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Laboratory Of Development Systems
 *
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
 *
 * Author: Carlos Lopes <carloslopesufpa@gmail.com>
 */

#include "ns3/network-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/position-allocator.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/config-store.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/propagation-module.h"
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/wifi-module.h"
#include <time.h>
#include "ns3/gnuplot.h"
#include "ns3/basic-energy-source.h"
#include "ns3/simple-device-energy-model.h"
#include <fstream>
#include "ns3/core-module.h"
#include "ns3/command-line.h"
#include "ns3/wifi-phy.h"
#include <stdlib.h>
#include <math.h>
using namespace ns3;

//Criação das Matrizes
double** create(int rows, int columns){
    double** table = new double*[rows];
    for(int i = 0; i < rows; i++) {
        table[i] = new double[columns]; 
        for(int j = 0; j < columns; j++){ 
        table[i][j] = 0; 
        }// sample set value;    
    }
    return table;
}
NS_LOG_COMPONENT_DEFINE ("Wifi_Alg");



int cenario = 11;

int main (int argc, char *argv[]) {

//Configurações da rede
    // Novo Retransmissor
    int rn = 3;
    // Total de usuários da rede
    int nAll = 50; 
    std::string nall = std::to_string(nAll);
    int nRn = cenario;
  
    int Sub = 1;
    std::string sub = std::to_string(Sub);
    // Numero de Clientes NÃO RETRANSMISSORES
    int nCli = nAll - nRn;
    // Numero de nós previamente conectados
    int nCon = 28;
    // Vetor com todos os Retransmissores
    int vet[nRn][1] = {rn, 13, 44, 43, 0, 5, 23, 47, 14, 28, 38};
    // Vetor com clientes previamente instalados
    int cli[nCon][1] = {3, 5, 14, 19, 22, 23, 41, 42, 43, 46, 1, 27, 29, 33, 40, 47, 0, 8, 20, 49, 2, 12, 28, 4, 38, 11, 16, 48};
    // Vetor com Retransmissores previamente instalados
    int ser[nCon][1] = {13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 44, 44, 44, 44, 44, 44, 43, 5, 5, 5, 47, 47, 47, 28, 28, 38, 38, 38};
 

    double simTime = 1200;
    // Tamanho máximo de Pacotes
    uint32_t MaxPacketSize = 300;
    // Intervalo de envio de pacotes
    double PacketInterval = 0.1;
    // Quantidade de Quadrantes
    int nQd = 1;
    std::string qd = std::to_string(nQd);
    //Tamanho da Grade da Simulação 
    uint16_t grid = 500;
    std::string gr = std::to_string(grid);
    // Tipos de Mobilidade
    bool constant = true;
    bool manual = true;
    
//Variáveis para receber dados do FlowMonitor
    int col = 1;
    double** Vazao = create(nAll, col);
    double** Atraso = create(nAll, col);
    double** Loss = create(nAll, col);
    double** Jitter = create(nAll, col);
    double atraso1 = 0;
    double atraso2 = 0;
    double** Energia = create(nAll, col);
    int aux_energy = 0;


//Obter posição de arquivo .txt
  double x[nAll][1];
  double y[nAll][1];
  FILE *arqx;
  FILE *arqy;
  char Linhax[nAll];
  char Linhay[nAll];
  char *xAux;
  char *yAux;
  char *resultx;
  char *resulty;
  // Abre um arquivo TEXTO para LEITURA
  arqx = fopen("Master_Node/coordenadas/quadrante_1/1_1_50_pos_x.txt", "rt"); 
  arqy = fopen("Master_Node/coordenadas/quadrante_1/1_1_50_pos_y.txt", "rt");
  if (arqx == NULL || arqy == NULL)  // Se houve erro na abertura
  {
     printf("Problemas na abertura do arquivo\n");
     return(0);
  }
  for(int i = 0; i<nAll; ++i){
      // Lê uma linha (inclusive com o '\n')
      resultx = fgets(Linhax, nAll, arqx);  // Ler os caracteres ou até '\n'
      if (resultx)  // Se foi possível ler
      xAux = Linhax;
      x[i][0] = atof(xAux);

      // Lê uma linha (inclusive com o '\n')
      resulty = fgets(Linhay, nAll, arqy);  // Ler os caracteres ou até '\n'
      if (resulty)  // Se foi possível ler
      yAux = Linhay;
      y[i][0] = atof(yAux);
  }




//Criando Nós da Simulação
// Criando Grupo de todos os nós
  NodeContainer wifiAll;
  wifiAll.Create (nAll);

//Setar Mobilidade dos nós
  MobilityHelper mobilitywifiAll;
  if(manual == false){
    mobilitywifiAll.SetPositionAllocator ("ns3::RandomRectanglePositionAllocator",
                                "X", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max="+gr+"]"),
                                "Y", StringValue ("ns3::UniformRandomVariable[Min=0.0|Max="+gr+"]"));
  }

  if(constant == true){
  //Constante Mobillity
    mobilitywifiAll.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  }

  if(constant == false){
  //Random Mobillity
    mobilitywifiAll.SetMobilityModel ("ns3::RandomWalk2dMobilityModel",
                             "Speed", StringValue ("ns3::ConstantRandomVariable[Constant=100.0]"),
                             "Bounds", StringValue ("-"+gr+"|"+gr+"|-"+gr+"|"+gr+""));
  }
  mobilitywifiAll.Install (wifiAll);

  if(manual == true){   
    for(int i = 0; i < nAll; ++i){ 
        wifiAll.Get (i)->GetObject<MobilityModel> ()->SetPosition (Vector (x[i][0], y[i][0], 0));
     } 
  }     
//Configuração da rede
  std::string phyMode ("ErpOfdmRate54Mbps");
  bool verbose = false;

// The below set of helpers will help us to put together the wifi NICs we want
  WifiHelper wifi;
  if (verbose)
    {
      wifi.EnableLogComponents ();  // Turn on all Wifi logging
    }

  YansWifiPhyHelper wifiPhy =  YansWifiPhyHelper::Default ();
// set it to zero; otherwise, gain will be added
  wifiPhy.Set ("RxGain", DoubleValue (0) );
// ns-3 supports RadioTap and Prism tracing extensions for 802.11b
  wifiPhy.SetPcapDataLinkType (YansWifiPhyHelper::DLT_IEEE802_11_RADIO);

// Criação do modelo de propagação da rede
  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  wifiChannel.AddPropagationLoss ("ns3::FriisPropagationLossModel");

  wifiPhy.SetChannel (wifiChannel.Create ());

// Adição do mac
  WifiMacHelper wifiMac;
  wifi.SetStandard (WIFI_PHY_STANDARD_80211g);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode",StringValue (phyMode),
                                "ControlMode",StringValue (phyMode));
  
NetDeviceContainer allDevice;

//AD HOC - NETWORK
 
  wifiMac.SetType ("ns3::AdhocWifiMac");
  allDevice = wifi.Install (wifiPhy, wifiMac, wifiAll);



//Atribuindo energia nos dispositivos
      srand((unsigned)time(0));
      for (int l=0; l<nAll; ++l){
          aux_energy = rand()%((100-50) + 50); 

          Ptr<BasicEnergySource> energySource = CreateObject<BasicEnergySource>();
          Ptr<SimpleDeviceEnergyModel> energyModel = CreateObject<SimpleDeviceEnergyModel>();

          energySource->SetInitialEnergy (aux_energy);
          energyModel->SetEnergySource (energySource);
          energySource->AppendDeviceEnergyModel (energyModel);
          energyModel->SetCurrentA (20);

          // aggregate energy source to node
          wifiAll.Get(l)->AggregateObject (energySource);
          // energy[l][0] = energySource;
          Energia[l][0] = aux_energy;
        }

//Intalação da pilha TCP/IP nos nós
  InternetStackHelper internet;
  internet.Install (wifiAll);
    
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer allInterface = ipv4.Assign (allDevice);
  
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

// Criação da aplicação UDP
// Aplicação para rede AP/STA
  
uint16_t port = 4000;
  std::string ipAp[nRn][1];
  int aux = 0;
  bool first = false;
  
  // int l = 1;
  bool entra = true;
  ApplicationContainer apps;
// Instalação dos Relay Nodes (RN)
                   
    for(int s = 0; s < nRn; ++s){
      UdpServerHelper server (port);
      server.Install (wifiAll.Get(vet[s][0]));  
      apps.Start (Seconds (0.1));
      apps.Stop (Seconds (simTime));
      aux = vet[s][0] + 1;
      ipAp[s][0] = "192.168.1." + std::to_string(aux);
      std::cout<<"IP = " << ipAp[s][0] <<std::endl;
    }
  
  for(int p = 0; p<nAll; ++p)
  {
          entra = true;  
        if(first == false){
          for(int s = 1; s<nRn; ++s)
          {
              for(int x = 0; x<nCon; ++x)
              {
                  if(p == cli[x][0] && ser[x][0] == vet[s][0])
                  {
                      entra=false;
                      UdpClientHelper client (Ipv4Address (ipAp[s][0].c_str()), port); 
                      client.SetAttribute ("MaxPackets", UintegerValue ((uint32_t)(simTime*(1/PacketInterval))));
                      client.SetAttribute ("Interval", TimeValue (Seconds (PacketInterval)));
                      client.SetAttribute ("PacketSize", UintegerValue (MaxPacketSize));
                      
                      // Instalar Aplicação em todos os nós Usuários
                      apps = client.Install (wifiAll.Get(p));
                      apps.Start (Seconds (0.1));
                      apps.Stop (Seconds (simTime));   
                      std::cout<<"Padrão:    Client = " << p <<"    Server = " << ipAp[s][0] <<std::endl;
                    
                  }
              }
          }
        }
          if(entra == true){
              if(p != rn && p != 13 && p != 44 && p != 43 && p != 0 && p != 5 && p != 23 && p != 47 && p != 14 && p != 28 && p != 38)
              {
                  //Configuração da aplicação   
                  UdpClientHelper client (Ipv4Address (ipAp[0][0].c_str()), port); 
                  client.SetAttribute ("MaxPackets", UintegerValue ((uint32_t)(simTime*(1/PacketInterval))));
                  client.SetAttribute ("Interval", TimeValue (Seconds (PacketInterval)));
                  client.SetAttribute ("PacketSize", UintegerValue (MaxPacketSize));

                  std::cout<<"Client = " << p <<"    Server = " << ipAp[0][0] <<std::endl;
                  
                  // Instalar Aplicação em todos os nós Usuários
                  apps = client.Install (wifiAll.Get(p));
                  apps.Start (Seconds (0.1));
                  apps.Stop (Seconds (simTime)); 
                  
              }
            }
  }
  
//FLOW-MONITOR
//Intalar FlowMonitor em todos os Nós
  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll ();
  std::string gp = std::to_string(cenario);

//Gerar animação
        AnimationInterface anim ("Master_Node/algoritmo/1_1_" + gp + "_master_node_alg.xml"); // Mandatory
        
        for (int i = 0; i < nCli; ++i){
          anim.UpdateNodeDescription (wifiAll.Get(i), "Node"); // Optional
          anim.UpdateNodeColor (wifiAll.Get(i), 255, 0, 0); // Coloração
        }
        
        for (int i = 0; i<nRn; ++i){
            anim.UpdateNodeDescription (wifiAll.Get(vet[i][0]), "RN"); // Optional
            anim.UpdateNodeColor (wifiAll.Get(vet[i][0]), 255, 255, 255); // Coloração
        }
        
        anim.EnablePacketMetadata (); // Optiona
        anim.EnableIpv4RouteTracking ("Master_Node/algoritmo/1_1_" + gp + "_master_node_alg_route.xml", Seconds (0), Seconds (5), Seconds (0.25)); //Optional
        anim.EnableWifiMacCounters (Seconds (0), Seconds (simTime)); //Optional
        anim.EnableWifiPhyCounters (Seconds (0), Seconds (simTime)); //Optional
  Simulator::Stop(Seconds(simTime));
  Simulator::Run();
 
//Imprimir resultados da simulação
  monitor->CheckForLostPackets ();
  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
  FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats ();

int u = 0;
int cont = 0;
double nsumLoss = 0;
double nmedLoss = 0;

double sumThroughput = 0;
double sumLoss = 0;
double sumAtraso = 0;
double sumJitter = 0;

double medThroughput = 0;
double medLoss = 0;
double medAtraso = 0;
double medJitter = 0;
double dur = 0;
int install[nAll][1];

for(int x = 0; x< nAll; ++x){
install[x][0] = 0;
}

std::string ipc;
std::string ips;

  for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin (); i != stats.end (); ++i)
  {
      Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (i->first);
      dur = (i->second.timeLastRxPacket.GetSeconds()-i->second.timeFirstTxPacket.GetSeconds());
      if(dur > 0)
      {        
          for(int s = 0; s<nRn; ++s)
            { 
               if (t.destinationAddress == ipAp[s][0].c_str())
               {                    
                    std::cout << "Flow " << i->first << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\n";
                    std::cout<<"Duration  : "<< dur <<std::endl;
                    std::cout<<"Tx Packets = " << i->second.txPackets<<std::endl;
                    std::cout<<"Rx Packets = " << i->second.rxPackets<<std::endl;
                    
                    Loss[u][0] = i->second.txPackets - i->second.rxPackets;
                    std::cout << "Perda de Pacotes: "<< Loss[u][0]<<std::endl;
                    
                    Vazao[u][0] = i->second.rxBytes * 8.0 / (i->second.timeLastRxPacket.GetSeconds()-i->second.timeFirstTxPacket.GetSeconds())/1024;
                    std::cout << "Vazão: " << Vazao[u][0] << " Kbps\n";

                    std::cout << "Energia: "<< Energia[u][0] <<std::endl;

                    Atraso[u][0] = ((i->second.timeLastRxPacket.GetSeconds()) - (i->second.timeLastTxPacket.GetSeconds()));
                    std::cout << "Atraso: "<< Atraso[u][0] <<std::endl;

                    atraso2 = i->second.timeLastRxPacket.GetSeconds()-i->second.timeLastTxPacket.GetSeconds();
                    atraso1 = i->second.timeFirstRxPacket.GetSeconds()-i->second.timeFirstTxPacket.GetSeconds();
                    Jitter[u][0] = atraso2 - atraso1;
                    std::cout << "Jitter: "<< Jitter[u][0] <<std::endl;
                    
                    sumLoss = sumLoss + Loss[u][0];
                    sumThroughput = sumThroughput + Vazao[u][0];
                    sumAtraso = sumAtraso + Atraso[u][0];
                    sumJitter = sumJitter + Jitter[u][0];
                    std::cout << " " <<std::endl;
                    
                if (t.destinationAddress == ipAp[0][0].c_str()){
                 for(int j = 0; j<nAll; ++j){
                    ipc = "192.168.1." + std::to_string(j+1);
                    if(t.sourceAddress == ipc.c_str()){
                      install[u][0] = j;
                      std::cout << "Client "<< install[u][0] <<std::endl;
                      break;
                    }else{
                      install[u][0] = 999;
                    }
                  }
                }

                    u++;
                }
          }
      }else{
        cont++;
        nsumLoss = nsumLoss + i->second.txPackets;
      }
  }

  
  
// Obter Média da Simulação
   medLoss = sumLoss / (u+1);
   nmedLoss = (nsumLoss+sumLoss)/nCli;
   medThroughput = sumThroughput / (u+1);
  
   medAtraso = sumAtraso / (u+1);
   medJitter = sumJitter / (u+1);

 double sumDeVazao = 0;
 double sumDeLoss = 0;
 double sumDeAtraso = 0;
 double sumDeJitter = 0;
 double VarVazao = 0;
 double VarLoss = 0;
 double VarAtraso = 0;
 double VarJitter  = 0;
 double DesVazao = 0;
 double DesLoss = 0;
 double DesAtraso  = 0;
 double DesJitter = 0;



      for(int y = 0; y<u; ++y){
        sumDeVazao = sumDeVazao + (Vazao[y][0]-medThroughput)*(Vazao[y][0]-medThroughput);
        sumDeLoss = sumDeLoss + (Loss[y][0]-medLoss)*(Loss[y][0]-medLoss);
        sumDeAtraso = sumDeAtraso + (Atraso[y][0]-medAtraso)*(Atraso[y][0]-medAtraso);
        sumDeJitter = sumDeJitter + (Jitter[y][0]-medJitter)*(Jitter[y][0]-medJitter);
      }
      VarVazao = sumDeVazao/u;
      VarLoss = sumDeLoss/u;
      VarAtraso = sumDeAtraso/u;
      VarJitter = sumDeJitter/u;
      
      DesVazao = sqrt(VarVazao);
      DesLoss = sqrt(VarLoss);
      DesAtraso = sqrt(VarAtraso);
      DesJitter = sqrt(VarJitter);

   std::cout << " " <<std::endl;
   std::cout << "Média Loss(Conectados): "<< medLoss <<std::endl;
   std::cout << "Média Loss(Total): "<< nmedLoss <<std::endl;
   std::cout << "Média Vazão: "<< medThroughput <<std::endl;
   std::cout << "Média Atraso: "<< medAtraso <<std::endl;
   std::cout << "Média Jitter: "<< medJitter <<std::endl;
   
   std::cout << " " <<std::endl;

   std::cout << "Desvio Padrão Loss(Conectados): " << DesLoss <<std::endl;
   std::cout << "Desvio Padrão Vazão: " << DesVazao <<std::endl;
   std::cout << "Desvio Padrão Atraso: " << DesAtraso <<std::endl;
   std::cout << "Desvio Padrão Jitter: " << DesJitter <<std::endl;


  std::cout << " " <<std::endl;
  std::cout << "Usuários Não Cobertos: "<< cont <<std::endl;
  std::cout << " " <<std::endl;
  std::cout << "Usuários Cobertos "<< u <<std::endl;
  std::cout << " " <<std::endl;
  std::cout << "Quantidade de Retransmissores "<< nRn <<std::endl;
  std::cout << " " <<std::endl;
  std::cout << "Nó Selecionado "<< vet[0][0] <<std::endl;


    std::ofstream myfile ("Master_Node/algoritmo/1_1_"+gp+"_Results_Algoritmo.csv");
     
 if (myfile.is_open())
  {
      myfile << "Nó Selecionado, Qt. Nós Cobertos, Média Loss(Conectados), DP Loss(Conectados), Perda de Pacotes (%), Média Loss(Total), Média Vazão, DP Vazão, Média Atraso, DP Atraso, Média Jitter, DPJitter, PDR\n";
      myfile << std::to_string(vet[0][0])+", "+std::to_string(u)+", "+std::to_string(medLoss)+", "+std::to_string(DesLoss)+", "+std::to_string(((medLoss/11999)*100))+", "+std::to_string(nmedLoss)+","+std::to_string(medThroughput)+","+std::to_string(DesVazao)+", "+std::to_string(medAtraso)+", "+std::to_string(DesAtraso)+", "+std::to_string(medJitter)+","+std::to_string(DesJitter)+", "+std::to_string(((11999-medLoss)/11999)*100)+"\n";
      myfile.close();
  }
  else std::cout << "Unable to open file";

  std::ofstream con ("Master_Node/algoritmo/1_1_"+gp+"_Conectados_Alg.csv");

if (con.is_open())
  {
      con<<"Servidor "+std::to_string(rn)<<std::endl;
      con<<"Clientes"<<std::endl;

     for(int j = 0; j<=u; ++j){
        if(install[j][0] != 0.000000){
          con<< std::to_string(install[j][0]) <<std::endl;
          std::cout<< std::to_string(install[j][0]) <<std::endl;
        }
      }
     con.close();
  }
  else std::cout << "Unable to open file";


//LÓGICA DE SELEÇÃO

  Simulator::Destroy();
  return 0;

}