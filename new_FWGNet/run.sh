#!/usr/bin/env bash

# Setting source code directory
src="/home/carl/FWGNet/new_FWGNet/"
# # Set variable start capturing of file
const_Size="True"
# Tipo de tamanho de pacote: "const_Value"(Valor específico) | "mean_Trace"(Usa a média do tamanho dos pacotes do trace)
type_Size="const_Value"
save_Graph="True"
case_Study="p2p"
# "99.80%";"99.85%";"99.90%";"99.95%";"99.99%"
IC="95.0"

get_pcap="False"
interface="wlp61s0" 

str_n_param="3"
str_n_row="3"
str_n_file_param="2"
str_verbose="True"
rate="4096kb/s"
lat="0.2ms"

# Capturing traffic (If get_pcap == "True")
if [[ $get_pcap = "True" ]]
then
    export interface
    # Start capture traffic module
    ./Shell/capture.sh
    echo "End of Capture Module"
fi
# # If get_pcap == "False" get a exist file
if [[ $get_pcap = "False" ]]
then
    # .pcap file directory
    dir_pcap_file="/media/carl/95d90125-d7fb-4efd-8875-ceead818ba80/Traces/Case-Study-1/small-tcp-http.pcap"
    export dir_pcap_file
    filter_pcap="real"
    export filter_pcap
    export case_Study
    # Start filter module (real .pcap file)
    ./Shell/filter.sh
    echo "End of Real Filter Module"
fi


# Start workload generation module


python Python/workload_generation.py "$const_Size" "$type_Size" "$save_Graph" "$case_Study" "$IC"
echo "End of Workload Generation Module"

# Start NS3 module 
dir_NS3="/home/carl/repos/ns-3-allinone/ns-3.31/"
file_Simulation="wifi-simple-adhoc"
# file_Simulation="oriented_object_simulation"
file_Simulation="eth"
file_Myapp="myapp.h"
file_Dir="/home/carl/FWGNet/new_FWGNet/C++/"


# Copy file to scratch
sudo cp ${file_Dir}${file_Simulation}.cc ${dir_NS3}scratch
# # Set permissions to file in scratch
sudo chmod 777 ${dir_NS3}scratch/${file_Simulation}.cc

# Copy file to scratch
sudo cp ${file_Dir}${file_Myapp} ${dir_NS3}
# # Set permissions to file in scratch
sudo chmod 777 ${dir_NS3}${file_Myapp}

# Open NS3 directory
cd ${dir_NS3} 

# # Run simulation file ########################################################################

# sudo ./waf --run "scratch/${file_Simulation} --str_n_param=${str_n_param}"
sudo ./waf --run "scratch/${file_Simulation} --str_n_param=${str_n_param} --str_n_row=${str_n_row} --str_n_file_param=${str_n_file_param} --str_verbose=${str_verbose} --rate=${rate} --lat=${lat}" # --phyMode=DsssRate2Mbps --rss=-50 --packetSize=500 --numPackets=100 --interval=2 "
# Debbuger
# sudo ./waf --run scratch/${file_Simulation} --command-template="g++ %s"

sudo rm ${dir_NS3}scratch/${file_Simulation}.cc
sudo rm ${dir_NS3}${file_Myapp}

echo "End of Simulation Module"

# Open source directory
cd ${src}
# Start filter module (simulated .pcap file)
# .pcap file directory
dir_pcap_file="/home/carl/repos/ns-3-allinone/ns-3.31/huehuehue-0-0.pcap"
export dir_pcap_file
# Start filter module (real .pcap file)
filter_pcap="ns3"
export filter_pcap

./Shell/filter.sh
echo "End of Simulated Filter Module"
# Start compare module
python Python/compare.py "$const_Size" "$type_Size" "$save_Graph" "$case_Study" "$IC"
echo "End of Compare Module"