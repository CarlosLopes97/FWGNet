#!/usr/bin/env bash

# Setting source code directory
src="/home/carl/FWGNet/new_FWGNet/"
# # Set variable start capturing of file
get_pcap="False"

# Capturing traffic (If get_pcap == "True")
if [[ $get_pcap = "True" ]]
then
    # Start capture traffic module
    # ./Shell/capture.sh
    echo "End of Capture Module"
fi
# # If get_pcap == "False" get a exist file
if [[ $get_pcap = "False" ]]
then
    # .pcap file directory
    dir_pcap_file="/media/carl/95d90125-d7fb-4efd-8875-ceead818ba80/Traces/Case-Study-1/small-tcp-http.pcap"
    export dir_pcap_file
    # Start filter module (real .pcap file)
    ./Shell/filter.sh
    echo "End of Real Filter Module"
fi


# Start workload generation module
python Python/workload_generation.py
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

# # Run simulation file
sudo ./waf --run "scratch/${file_Simulation}" # --phyMode=DsssRate2Mbps --rss=-50 --packetSize=500 --numPackets=100 --interval=2 --verbose=true"
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
./Shell/filter.sh
# echo "End of Simulated Filter Module"
# Start compare module

echo "End of Compare Module"