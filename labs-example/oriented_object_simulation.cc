/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software m_foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * m_foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Autor: Hygor Jardim da Silva, Castanhal - Pará, Brasil.
 * Contato: hygorjardim@gmail.com - https://hygorjardim.github.io/
 * Universidade Federal do Pará - Campus Universitário de Castanhal
 * Faculdade de Computação - FACOMP
 * Laboratório de Desenvolvimento de Sistemas - LADES
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"
#include "ns3/config-store-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/ns2-mobility-helper.h"
#include "ns3/gnuplot.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/csma-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/position-allocator.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/config-store.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/propagation-module.h"
#include "ns3/netanim-module.h"
#include "ns3/basic-energy-source.h"
#include "ns3/simple-device-energy-model.h"
#include "ns3/command-line.h"
#include "ns3/wifi-phy.h"
// #include "ns3/aodv-module.h"
// #include "ns3/olsr-module.h"
// #include "ns3/dsdv-module.h"
// #include "ns3/gpsr-module.h"
// #include "ns3/dsr-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("SimulationFramework");
std::string** create_mat(int rows, int columns)
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

std::string* create_arr(int rows)
{
	std::string* table = new std::string[rows];
	
	for(int i = 0; i < rows; i++) 
	{
		// table[i] = new std::string[columns]; 
		// for(int j = 0; j < columns; j++)
		// { 
		// table[i] = std::string("0"); 
		table[i] = "0"; 
		// std::cout<<table[i]<<std::endl;
		// }// sample set value;    
	}
	return table;
}

void Throughput (FlowMonitorHelper *fmhelper, Ptr<FlowMonitor> flowMon,Gnuplot2dDataset DataSet, std::string name)
{
	double localThrou=0;
	std::map<FlowId, FlowMonitor::FlowStats> flowStats = flowMon->GetFlowStats();
	Ptr<Ipv4FlowClassifier> classing = DynamicCast<Ipv4FlowClassifier> (fmhelper->GetClassifier());
	for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator stats = flowStats.begin (); stats != flowStats.end (); ++stats)
	{
		if (stats->first == 2)
		{
			//Ipv4FlowClassifier::FiveTuple fiveTuple = classing->FindFlow (stats->first);
			localThrou=(stats->second.rxBytes * 8.0 / (stats->second.timeLastRxPacket.GetSeconds()-stats->second.timeFirstTxPacket.GetSeconds())/1024/1024);
			DataSet.Add((double)Simulator::Now().GetSeconds(),(double) localThrou);
		}
	}
	Simulator::Schedule(Seconds(1), &Throughput, fmhelper, flowMon, DataSet, name);
	{
    	flowMon->SerializeToXmlFile (name + "-throughput.xml", true, true);
	}
}
void Jitter (FlowMonitorHelper *fmHelper, Ptr<FlowMonitor> flowMon, Gnuplot2dDataset Dataset2, std::string name)
{
	double localJitter = 0;
	double atraso1 = 0;
	double atraso2 = 0;
	std::map<FlowId, FlowMonitor::FlowStats> flowStats2 = flowMon->GetFlowStats();
	Ptr<Ipv4FlowClassifier> classing2 = DynamicCast<Ipv4FlowClassifier> (fmHelper->GetClassifier());
	for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator stats2 = flowStats2.begin(); stats2 != flowStats2.end(); ++stats2)
	{
		if(stats2->first == 2)
		{
			//Ipv4FlowClassifier::FiveTuple fiveTuple2 = classing2->FindFlow (stats2->first);
			atraso2 = stats2->second.timeLastRxPacket.GetSeconds()-stats2->second.timeLastTxPacket.GetSeconds();
			atraso1 = stats2->second.timeFirstRxPacket.GetSeconds()-stats2->second.timeFirstTxPacket.GetSeconds();
			localJitter= atraso2-atraso1;
			Dataset2.Add((double)Simulator::Now().GetSeconds(), (double) localJitter);
		}

	atraso1 = atraso2;
	}

	Simulator::Schedule(Seconds(1),&Jitter, fmHelper, flowMon, Dataset2, name);
	{
		flowMon->SerializeToXmlFile (name + "-jitter.xml", true, true);
	}
}
void Delay (FlowMonitorHelper *fmHelper, Ptr<FlowMonitor> flowMon, Gnuplot2dDataset Dataset3, std::string name)
{
	double localDelay = 0;
	std::map<FlowId, FlowMonitor::FlowStats> flowStats3 = flowMon->GetFlowStats();
	Ptr<Ipv4FlowClassifier> classing3 = DynamicCast<Ipv4FlowClassifier> (fmHelper->GetClassifier());
	for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator stats3 = flowStats3.begin(); stats3 != flowStats3.end(); ++stats3)
	{
		if(stats3->first == 2)
		{
			//Ipv4FlowClassifier::FiveTuple fiveTuple3 = classing3->FindFlow (stats3->first);
			localDelay = stats3->second.timeLastTxPacket.GetSeconds()-stats3->second.timeLastRxPacket.GetSeconds();
			Dataset3.Add((double)Simulator::Now().GetSeconds(), (double) localDelay);
		}
  	}
	Simulator::Schedule(Seconds(1),&Delay, fmHelper, flowMon, Dataset3, name);
	{
  		flowMon->SerializeToXmlFile (name + "-delay.xml", true, true);
	}
}

class Simulation
	{
	public:
		Simulation (std::string name);

		void Configure (int argc, char ** argv);

		int Run ();

		void Report (std::ostream & os);

		void GraphicPlot ();

		void GraphicClose ();

		void SetProtocol (int p);

		void SetSimulationTime (double st);

		void InstallApplication ();

		// Additional Function
		void ReadFiles ();

		void CreateTrace ();

		// Private
		void CreateNodes ();

		void Mobility ();

		void InstallInternetStack();

		// void ReceivePacket ();

		// void GenerateTraffic ();

	private:

		uint32_t 	m_nodes;
		uint32_t  	m_sinkNode;
		int 		m_xSize;
		int 		m_ySize;
		double 		m_step;
		double  	m_packetInterval;
		uint32_t    m_protocol; // 1 = AODV, 2 = OLSR, 3 = DSDV, for future DSR and GPSR
		uint32_t 	m_application; // 1 = OnOff, 2 = EcoSever
		uint32_t 	m_packetsSize;
		uint32_t 	m_maxPackets;
		double 		m_simulationTime;
		std::string m_phyMode;
		std::string m_simulationName;
		uint32_t  m_mobilitymodel; // 1 = Grid, 2 = Ns2Mobility/SUMO
		std::string m_traceFile;
		
		// Additional Variables
		double m_rss;
		uint32_t m_packetSize;
		uint32_t m_numPackets;
		double m_interval;
		Time m_interPacketInterval;
		bool m_verbose;
		
		int m_n_nodes;

		int m_n_file;
		int m_n_param;

		std::string* arr_m_device;
		std::string* arr_m_container;
		std::string* arr_m_stack;
		std::string* arr_m_adderess;
		
		
		std::string** m_proto_ip;
		std::size_t m_found;
		
		PointToPointHelper p2p;
		WifiHelper wifi;
		CsmaHelper csma;
		YansWifiPhyHelper wifiPhy;
		YansWifiChannelHelper wifiChannel;

		NodeContainer m_container;
		NetDeviceContainer m_device;
		InternetStackHelper m_internetStack;
		
		MobilityHelper mobility;

		Ipv4AddressHelper m_ipv4;
		Ipv4InterfaceContainer m_adderess;

		ApplicationContainer serverApps;
		
		Gnuplot gnuplot, gnuplot2, gnuplot3;
		Gnuplot2dDataset dataset, dataset2, dataset3;
		
		FlowMonitorHelper fmhelper;
		Ptr<FlowMonitor> allMon;
		
	// private:

	

	};

	Simulation::Simulation (std::string name):
		m_nodes (25),
		m_sinkNode (12),
		m_xSize (20),
		m_ySize (20),
		m_step (5),
		m_packetInterval (0.01),
		m_protocol (1),
		m_application (1),
		m_packetsSize (1024),
		m_maxPackets (10000),
		m_simulationTime (2.0),
		m_phyMode ("DsssRate11Mbps"),
		m_simulationName (name),
		m_mobilitymodel (1),
		m_traceFile("mobility/mobility.tcl​"),
		
		// Additional variables
		m_rss(-60),
		m_numPackets (100),
		m_interval (2.0), // seconds
		m_interPacketInterval (Seconds (m_interval)),
		m_verbose (false),
		
		m_n_nodes (2), 
		
		m_n_file (2),
		m_n_param (3),

		arr_m_device (create_arr(m_n_file)),
		arr_m_container (create_arr(m_n_file)),
		arr_m_stack (create_arr(m_n_file)),
		arr_m_adderess (create_arr(m_n_file)),

		m_proto_ip (create_mat(m_n_file, m_n_param)),
		m_found (0),

		m_container ("eth"),
		m_device ("node")

	
	{
	}
	void Simulation::SetSimulationTime (double st)
	{
		m_simulationTime = st;
	}
	void Simulation::SetProtocol (int p)
	{
		m_protocol = p;
		
		// for (int i = 0; i<m_n_file; ++i){
		// 	for (int j = 0; j<m_n_param; ++j){
		
		// 	m_proto_ip[i][0]
			
		// 	}	
		// }

	}
	void
	Simulation::Configure (int argc, char *argv[])
	{
		

		CommandLine cmd;

		cmd.AddValue ("phyMode", "Wifi Phy mode", m_phyMode);
		cmd.AddValue ("rss", "received signal strength", m_rss);
		cmd.AddValue ("packetSize", "size of application packet sent", m_packetSize);
		cmd.AddValue ("numPackets", "number of packets generated", m_numPackets);
		cmd.AddValue ("interval", "interval (seconds) between packets", m_interval);
		cmd.AddValue ("verbose", "turn on all WifiNetDevice log components", m_verbose);

		cmd.Parse (argc, argv);
		m_interPacketInterval = Seconds (m_interval);
		LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
		// LogComponentEnable ("TcpEchoClientApplication", LOG_LEVEL_INFO);
	}
	void
	Simulation::CreateNodes ()
	{	
		std::string lat = "2ms";
  		std::string rate = "500kb/s"; // P2P link

		for (int i = 0; i<m_n_file; ++i)
		{
			m_n_nodes = stoi(m_proto_ip[i][1]) + stoi(m_proto_ip[i][2]);
			// std::cout<<"m_n_nodes: "<<m_n_nodes<<std::endl;
			// std::cout<<"m_n_nodes2: "<<<<std::endl;
			arr_m_container[i] = std::string("nodes_"+m_proto_ip[i][0]);
			m_container = arr_m_container[i];
			// NodeContainer m_container;
			m_container.Create (m_n_nodes);
			

			// NetDeviceContainer m_device;
			arr_m_device[i] = std::string("devs_" + m_proto_ip[i][0]);
			m_device = arr_m_device[i];
			
			m_found = m_proto_ip[i][0].find("eth");
			
			// int m_n_nodes = stoi(m_proto_ip[i][j]);
			// m_container = std::string(m_proto_ip[i][0]);
			// NetDeviceContainer m_device;
			if (m_found!=std::string::npos && m_n_nodes < 3)
			{
				// m_container.Create (m_n_nodes);
				p2p.SetDeviceAttribute ("DataRate", StringValue (rate));
				p2p.SetChannelAttribute ("Delay", StringValue (lat));
				m_device = p2p.Install (m_container);
			}
			if (m_found!=std::string::npos && m_n_nodes > 2)
			{	
				// NodeContainer m_container;
				csma.SetChannelAttribute ("DataRate", DataRateValue (DataRate (rate)));
				csma.SetChannelAttribute ("Delay", StringValue (lat));

				m_device = csma.Install (m_container);
				
			}
			m_found = m_proto_ip[i][0].find("radio");
			if (m_found!=std::string::npos)
			{	
				wifi.SetStandard(WIFI_PHY_STANDARD_80211b);
				wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
				wifiChannel.AddPropagationLoss ("ns3::FriisPropagationLossModel");
				wifiPhy.SetChannel (wifiChannel.Create ());

				WifiMacHelper wifiMac;
				wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
											"DataMode",StringValue (m_phyMode),
											"ControlMode",StringValue (m_phyMode));

				wifiMac.SetType ("ns3::AdhocWifiMac");
				m_device = wifi.Install (wifiPhy, wifiMac, m_container);
			}	
			// std::cout<<"m_device: "<<arr_m_device[i]<<std::endl;
			// std::cout<<"m_container: "<<arr_m_container[i]<<std::endl;
		}
	}
	void
	Simulation::Mobility ()
	{
		for (int i = 0; i<m_n_file; ++i)
		{
			if (m_mobilitymodel == 1) {
				mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
																			"MinX", DoubleValue (0.0), //onde inicia no eixo X
																			"MinY", DoubleValue (0.0), //onde inicia no eixo Y
																			"DeltaX", DoubleValue (m_xSize), // Distância entre nós
																			"DeltaY", DoubleValue (m_ySize), // Distância entre nós
																			"GridWidth", UintegerValue (m_step), // Quantidade de colunas em uma linha
																			"LayoutType", StringValue ("RowFirst")); // Definindo posições em linha
				mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
				

				m_container = arr_m_container[i];
				// NodeContainer m_container;
				// m_container.Create(m_n_nodes);

				
				mobility.Install (m_container);
				
			}

			// arr_m_device[i]
			// arr_m_container[i]


			// if (m_mobilitymodel == 2) {
			// 	Ns2MobilityHelper mobilityns2 = Ns2MobilityHelper (m_traceFile);
			// 	mobilityns2.Install ();
			// }
		}
	}
	

	void
	Simulation::InstallInternetStack ()
	{

	// 	AodvHelper aodv;
	// 	OlsrHelper olsr;
	// 	DsdvHelper dsdv;
	// 	GpsrHelper gpsr;
	// 	DsrHelper dsr;
	// 	DsrMainHelper dsrMain;
		// Ipv4ListRoutingHelper list;

		// InternetStackHelper m_internetStack;
	
	// 	switch (m_protocol)
	//     {
	//     case 1:
	//       list.Add (aodv, 10);
	//       break;
	//     case 2:
	//       list.Add (olsr, 10);
	//       break;
	//     case 3:
	//       list.Add (dsdv, 10);
	//       break;
	//     case 4:
	// 			list.Add (gpsr, 10);
	//       break;
	// 		case 5:
	// 		break;
	//     default:
	//       NS_FATAL_ERROR ("No such protocol:" << m_protocol);
	//     }

	// 	if (m_protocol < 5)
	// 	{
	// 	  m_internetStack.SetRoutingHelper (list);
	// 	  m_internetStack.Install (nodes);
	// 	}
	// 	else if (m_protocol == 5)
	// 	{
	// 		m_internetStack.Install (nodes);
    //   dsrMain.Install (dsr, nodes);
	// 	}
		// m_internetStack.SetRoutingHelper (list);
		// InternetStackHelper m_internetStack;
		// m_internetStack.Install ();
		for (int i = 0; i<m_n_file; ++i)
		{
			// arr_m_stack[i] = std::string("stack_"+m_proto_ip[i][0]);
			// std::cout<<"stack: "<<arr_m_stack[i]<<std::endl;
			// m_internetStack = arr_m_stack[i];

			// InternetStackHelper m_internetStack;

			
			
			
			m_container = arr_m_container[i];
			// NodeContainer m_container;
			// m_container.Create(m_n_nodes);
			
			m_internetStack.Install(m_container);
			
			Ipv4AddressHelper m_ipv4;
			
			std::string add;
			add = "10.1."+std::to_string(i+1)+".0";

			m_ipv4.SetBase (Ipv4Address(add.c_str()), "255.255.255.0");
			
			m_device = arr_m_device[i];
			// NetDeviceContainer m_device;
			arr_m_adderess[i] = add;
			// arr_m_adderess[i] = std::string("add_"+m_proto_ip[i][0]);
			// m_adderess = arr_m_adderess[i];
			
			Ipv4InterfaceContainer m_adderess = m_ipv4.Assign (m_device);
			
			// m_ipv4.Assign (m_device);
			Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

			// std::cout<<"container: "<<arr_m_container[i]<<std::endl;
			// std::cout<<"device: "<<arr_m_device[i]<<std::endl;
			// std::cout<<"stack: "<<arr_m_stack[i]<<std::endl;
			// std::cout<<"Add: "<<add<<std::endl;
			// arr_address[i] = m_ipv4;
		}

		// m_internetStack.Install (m_container);

		// Ipv4AddressHelper m_ipv4;
		// m_ipv4.SetBase ("10.1.1.0", "255.255.255.0");
		// interfaces = m_ipv4.Assign (nodesDevices);

	}
	
	void ReceivePacket (Ptr<Socket> socket)
	{
	while (socket->Recv ())
		{
		NS_LOG_UNCOND ("Received one packet!");
		}
	}

	static void GenerateTraffic (Ptr<Socket> socket, uint32_t pktSize, 
								uint32_t pktCount, Time pktInterval )
	{
	if (pktCount > 0)
		{
		socket->Send (Create<Packet> (pktSize));
		Simulator::Schedule (pktInterval, &GenerateTraffic, 
							socket, pktSize,pktCount-1, pktInterval);
		}
	else
		{
		socket->Close ();
		}
	}

	void
	Simulation::InstallApplication ()
	{
		for (int i = 0; i<m_n_file; ++i)
		{	
			// m_internetStack = arr_m_stack[i];
			// std::cout<<"stack: "<<arr_m_stack[i]<<std::endl;
			// InternetStackHelper m_internetStack;
			// arr_m_adderess[i] = std::string("add_"+m_proto_ip[i][0]);
			// m_adderess = arr_m_adderess[i];

			// Ipv4InterfaceContainer m_adderess;

			m_container = arr_m_container[i];
			// NodeContainer m_container;
			// m_container.Create(m_n_nodes);

			m_device = arr_m_device[i];
			// NetDeviceContainer m_device;

			UdpEchoServerHelper echoServer (9);
			serverApps = echoServer.Install (m_container.Get (0));
			serverApps.Start (Seconds (0.0));
			serverApps.Stop (Seconds (m_simulationTime));
											
			UdpEchoClientHelper echoClient (Ipv4Address (arr_m_adderess[i].c_str()), 9); 
			
			echoClient.SetAttribute ("MaxPackets", UintegerValue (m_maxPackets));
			echoClient.SetAttribute ("Interval", TimeValue (Seconds (m_packetInterval)));
			echoClient.SetAttribute ("PacketSize", UintegerValue (m_packetsSize));
			
			ApplicationContainer clientApps = echoClient.Install (m_container);
			clientApps.Start (Seconds (0.0));
			clientApps.Stop (Seconds (m_simulationTime));
			
	
	// TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
	// std::cout<<"10"<<std::endl;
	// Ptr<Socket> recvSink = Socket::CreateSocket (m_container.Get(1), tid);
	// std::cout<<"3"<<std::endl;
	// InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), 80);
	// std::cout<<"4"<<std::endl;
	// recvSink->Bind (local);
	// std::cout<<"5"<<std::endl;
	// recvSink->SetRecvCallback (MakeCallback (&ReceivePacket));
	// std::cout<<"6"<<std::endl;

	// Ptr<Socket> source = Socket::CreateSocket (m_container.Get(0), tid);
	// std::cout<<"7"<<std::endl;
	// InetSocketAddress remote = InetSocketAddress (Ipv4Address ("255.255.255.255"), 80);
	// std::cout<<"8"<<std::endl;
	// source->SetAllowBroadcast (true);
	// std::cout<<"9"<<std::endl;
	// source->Connect (remote);
	// std::cout<<"10"<<std::endl;

	// // Output what we are doing
	// NS_LOG_UNCOND ("Testing " << m_numPackets  << " packets sent with receiver rss " << m_rss);


	// Simulator::ScheduleWithContext (source->GetNode ()->GetId (),
	//                                 Seconds (1.0), &GenerateTraffic, 
	//                                 source, m_packetSize, m_numPackets, m_interPacketInterval);
		}
	// std::cout<<"OK"<<std::endl;
	}

	void
	Simulation::GraphicPlot ()
	{
		std::string graphicsFileName        = m_simulationName + "-throughput.png";
		std::string plotTitle               = "Flow vs Throughput";
		std::string dataTitle               = "Throughput";

		gnuplot.SetOutputFilename (graphicsFileName);
		gnuplot.SetTitle (plotTitle);
		gnuplot.SetTerminal ("png");
		gnuplot.SetLegend ("Flow", "Throughput");
		dataset.SetTitle (dataTitle);
		dataset.SetStyle (Gnuplot2dDataset::LINES_POINTS);
		allMon = fmhelper.InstallAll();

		Throughput (&fmhelper, allMon, dataset, m_simulationName);

		std::string graphicsFileName2        = m_simulationName + "-jitter.png";
		std::string plotTitle2               = "Flow vs Jitter";
		std::string dataTitle2       	     = "Jitter";

		gnuplot2.SetOutputFilename (graphicsFileName2);
		gnuplot2.SetTitle(plotTitle2);
		gnuplot2.SetTerminal("png");
		gnuplot2.SetLegend("Flow", "Jitter");
		dataset2.SetTitle(dataTitle2);
		dataset2.SetStyle(Gnuplot2dDataset::LINES_POINTS);

		Jitter(&fmhelper, allMon, dataset2, m_simulationName);

		std::string graphicsFileName3        = m_simulationName + "-delay.png";
		std::string plotTitle3               = "Flow vs Delay";
		std::string dataTitle3               = "Delay";

		gnuplot3.SetOutputFilename (graphicsFileName3);
		gnuplot3.SetTitle(plotTitle3);
		gnuplot3.SetTerminal("png");
		gnuplot3.SetLegend("Flow", "Delay");
		dataset3.SetTitle(dataTitle3);
		dataset3.SetStyle(Gnuplot2dDataset::LINES_POINTS);

		Delay(&fmhelper, allMon, dataset3, m_simulationName);
	}
	void
	Simulation::GraphicClose ()
	{
		std::string plotFileName = m_simulationName + "-throughput.plt";
		gnuplot.AddDataset (dataset);
		std::ofstream plotFile (plotFileName.c_str());
		gnuplot.GenerateOutput (plotFile);
		plotFile.close ();

		std::string plotFileName2 = m_simulationName + "-jitter.plt";
		gnuplot2.AddDataset(dataset2);
		std::ofstream plotFile2 (plotFileName2.c_str());
		gnuplot2.GenerateOutput(plotFile2);
		plotFile2.close();

		std::string plotFileName3 = m_simulationName + "-delay.plt";
		gnuplot3.AddDataset(dataset3);
		std::ofstream plotFile3 (plotFileName3.c_str());
		gnuplot3.GenerateOutput(plotFile3);
		plotFile3.close();
	}
	
	void Simulation::CreateTrace ()
	{
		std::cout<<"TRACE"<<std::endl;
		std::cout<<"OK"<<std::endl;
		for (int i = 0; i<m_n_file; ++i)
		{
			for (int j = 0; j<m_n_param; ++j)
			{	
				PcapHelper pcapHelper;
				std::cout<<"1"<<std::endl;
				m_found = m_proto_ip[i][0].find("eth");
				if (m_found!=std::string::npos && m_n_nodes < 3)
				{	
					std::cout<<"2"<<std::endl;
					// Tracing

					
					// Criando arquivo pcap
					Ptr<PcapFileWrapper> file = pcapHelper.CreateFile ("simulation.pcap", std::ios::out, PcapHelper::DLT_PPP);
					std::cout<<"2.1"<<std::endl;
					// p2p.EnablePcap ("simulation", arr_m_device[i]);
					std::cout<<"2.2"<<std::endl;
				}
				if (m_found!=std::string::npos && m_n_nodes >= 3)
				{	
					std::cout<<"3"<<std::endl;
					// Tracing
					// Criando arquivo pcap
					Ptr<PcapFileWrapper> file = pcapHelper.CreateFile ("simulation.pcap", std::ios::out, PcapHelper::DLT_PPP);
					// csma.EnablePcap ("simulation", arr_m_device[i]);
				}

				m_found = m_proto_ip[i][0].find("radio");
				if (m_found!=std::string::npos)
				{	
					std::cout<<"4"<<std::endl;
					// Tracing
					// Criando arquivo pcap
					Ptr<PcapFileWrapper> file = pcapHelper.CreateFile ("simulation.pcap", std::ios::out, PcapHelper::DLT_PPP);
					// wifiPhy.EnablePcap ("simulation", arr_m_device[i]);
				}
			}
		}
		std::cout<<"END_TRACE"<<std::endl;
	}
	int
	Simulation::Run ()
	{
		CreateNodes ();
		Mobility ();
		InstallInternetStack ();
		InstallApplication ();
		
		// GraphicPlot ();

		Simulator::Stop (Seconds (m_simulationTime));
		Simulator::Run ();
		// GraphicClose ();
		
		Simulator::Destroy ();
		CreateTrace ();
		return 0;
	}
	void
	Simulation::Report (std::ostream &)
	{
	}
	
	void
	Simulation::ReadFiles ()
	{
		// int m_n_file = 2;

		char *get_proto;
		int get_ip_src;
		int get_ip_dst;

		// int m_n_param = 3;
		// std::string** m_proto_ip = create_mat(m_n_file, m_n_param);
		
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
			while(fscanf(f, "%s %d %d", data1, &data2, &data3) == m_n_param) 
			{
				
				// printf("%s %d %d\n", data1, data2, data3);

				get_proto = data1;  
				if (get_proto)  
				m_proto_ip[i][0] = get_proto;
				
				get_ip_src = data2; 
				if (get_ip_src)  
				m_proto_ip[i][1] = std::to_string(get_ip_src);

				get_ip_dst = data3;
				if (get_ip_dst)  
				m_proto_ip[i][2] = std::to_string(get_ip_dst);

				i++;
			}
			fclose(f);
			

		}

		// for (int i = 0; i < m_n_file; ++i)
		// {
		// 	std::cout<<"PROTO: "<<m_proto_ip[i][0]<<" IP_SRC:"<<m_proto_ip[i][1]<<" IP_DST: "<<m_proto_ip[i][2]<<std::endl;
			
		// }
	}
int
main (int argc, char *argv[])
{	
	

	Simulation cenario ("cenario");
	cenario.ReadFiles();
	cenario.Configure (argc, argv);
	cenario.SetProtocol (1);
	cenario.SetSimulationTime (100);
	cenario.Run ();
	
}
