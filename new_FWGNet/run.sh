#!/usr/bin/env bash

# Setting source code directory
src="/home/carl/FWGNet/new_FWGNet/"
# Set variable start capturing of file
get_pcap="False"

# Capturing traffic (If get_pcap == "True")
if [[ $get_pcap = "True" ]]
then
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
    # Start filter module (real .pcap file)
    ./Shell/filter.sh
    echo "End of Real Filter Module"
fi


# Start workload generation module
python Python/workload_generation.py
echo "End of Workload Generation Module"

# Start NS3 module 
dir_NS3="/home/carl/repos/ns-3-allinone/ns-3.30/"
file_Simulation="simulation"


# Copy file to scratch
cp C++/${file_Simulation}.cc ${dir_NS3}scratch
# # Set permissions to file in scratch
sudo chmod 777 ${dir_NS3}scratch/${file_Simulation}.cc

# Open NS3 directory
cd ${dir_NS3} 

# # Run simulation file
# ./waf --run "scratch/${file_Simulation}"

echo "End of Simulation Module"

# Open source directory
cd ${src}
# Start filter module (simulated .pcap file)
./Shell/filter.sh
echo "End of Simulated Filter Module"
# Start compare module

echo "End of Compare Module"