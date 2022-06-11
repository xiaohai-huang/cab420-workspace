#!/bin/bash
full_name='Baorong_Huang'
python='python3.9'
# Question 1 output file
output_file="${full_name}_Q1.txt"
echo "Generating output for $output_file ..."
"$python" Q1.py Rnews_t120 > $output_file 

# Question 2 output file
output_file="${full_name}_Q2.txt"
echo "Generating output for $output_file ..."
"$python" Q2.py > "$output_file"
"$python" Q2.py "USA: RESEARCH ALERT - Minnesota Mining cut." >> $output_file
"$python" Q2.py "SOUTH AFRICA: Death toll reaches 24 in S.African mine clashes." >> $output_file
"$python" Q2.py "SOUTH AFRICA: Three killed in new clashes at S.Africa gold mine." >> $output_file

# Question 3 output file
output_file="${full_name}_Q3.txt"
echo "Generating output for $output_file ..."
echo -n "" > $output_file
"$python" Q3.py 'Deaths mining accidents' >> $output_file
"$python" Q3.py 'Mentioning deaths in mining accidents' >> $output_file
"$python" Q3.py 'Statistics on number of mining deaths' >> $output_file
"$python" Q3.py 'ethnic clashes and resultant deaths of mine workers near a mine' >> $output_file
                    
