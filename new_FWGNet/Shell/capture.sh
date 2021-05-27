#!/usr/bin/env bash

# Interface name
interface="wlp61s0" 
# Save directory and file name
dir_pcap_file="/tmp/log.pcap"
export dir_pcap_file

# Command of capturing traffic 
# More in wireshark Terminal Guide: https://www.wireshark.org/docs/wsug_html_chunked/ChCustCommandLine.html
# sudo tshark -w dir_pcap_file -i ${interface}
# sudo chmod 777 ${dir_pcap_file}

# Start filter module (real .pcap file)
./Shell/filter.sh

